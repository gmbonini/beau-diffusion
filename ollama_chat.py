# ============ Chat mínimo usando /api/chat com inputs já prontos ============

import json, requests, re
from loguru import logger


OLLAMA_URL = "http://127.0.0.1:11200"
MODEL = "qwen2.5vl:7b"

SYSTEM_CHAT = """
You are a prompt engineer and image reviewer for text-to-3D/multiview generation.

You will receive:
- an image avaliation (text),
- the current positive prompt (refined),
- the current negative prompt.
Avaliation:
- Compare the image quality to the prompts.
- You can be critical but not too much in your evaluation. If the images have minor issues but overall look good, say so. If they have major problems, point them out clearly. Only give constructive feedback.
- If the overall quality is good, you don't need to list every small issue and the negative prompt can be the same negative prompt received.
- The bigger problems are: geometry holes, extra limbs (if the prompt doesn't call for it), deformed face, artifacts, lighting, shadows, background.
- Example: if the input is "a cat", the images should look like a real cat from different angles. If the cat has extra limbs, deformed face, a extra tail, say so.

Behaviors:
- On START: return the three fields in the strict JSON format (you may lightly polish wording).
- On FEEDBACK: YOU MUST APPLY the client's requests by EDITING the prompts.
  - If the user asks to ADD or EMPHASIZE something > add concise tokens to the POSITIVE.
  - If the user asks to REDUCE/REMOVE or MITIGATE errors/artifacts > add/adjust constraints in the NEGATIVE (and optionally remove tokens from POSITIVE if conflicting).
  - If the user complain about something he saw, interpret it and add/remove/modify tokens accordingly on negative/positive. Example: it doesn't look natural > add "unnatural colors" to negative; it has too many trees > add "excessive trees" to negative; the dragon is not flying how i asked on the prompt > add "flying dragon" to positive and "grounded dragon" to negative.
  - Never return unchanged prompts when edits were requested.
- Never ignore the user's requests. Insert the changes based on the user's feedback and don't ignore the user's requests.
- Specify the chacteristics in the prompts using concise, specific tokens (no vague or generic terms). For example, if it says "cat with full orange body", specify "orange body", "orange members", "orange head", "orange tail", etc. But you don't need to be exhaustive.

Positive rules:
- Object-first (singular), concise attributes, preserve multi-word spatial relations; comma-separated.
- If the user wants to add something, add it to the positive prompt.
- Use concrete, visual tokens (avoid vague phrasing).
- Examples:
  * "more water" > add "dominant lake", "expanded water surface"
  * "increase detail on water surface" > add "detailed water surface"
  * "natural look" > add "natural colors", "moderate saturation"
  * "center the lake" > add "central lake", "lake in center"
  * "more green" > add "lush greenery", "vibrant foliage"

* "less trees" > remove objects or caracteristics if present; do NOT add to positive. Mitigate or remove in negative.

Negative rules:
- 6–8 items MAX (do not exceed); focus on artifacts/geometry/texture; comma-separated.
-If the user wants to mitigate/reduce/remove something, add it to the negative prompt.
- Map mitigation requests to negatives, examle.:
  * less trees > "excessive trees", "overcrowded foliage", "duplicate trees"
  * avoid repetitive foliage > "repetitive foliage", "tiling textures"
  * flat water > "flat water", "blocky water surface", "unrealistic reflections"
  * reduce saturation > "over-saturated colors"
  * remove background > keep "background" in negative
- ALWAYS append (even beyond the 8 cap): lighting, shadows, background.

- If the user complaint about something he saw or missed on the image, interpret it and add/remove/modify tokens accordingly on negative/positive.
Always have certain if a caracteristic should be in positive or negative. For example:
- "more trees" > add to positive
- "less trees" > mitigate/remove in negative (do NOT add to positive)
- "more detail" > add to positive
- "reduce detail" > mitigate/remove in negative (do NOT add to positive)
- "natural look" > add to positive
- "avoid artifacts" > add to negative

Conversation rule:
- Always include a short 1–2 sentence natural-language "message" addressing the user. On START: invite edits. On FEEDBACK: acknowledge changes and ask if more tweaks are needed. Keep it concise and friendly.

STRICT OUTPUT (JSON ONLY; no extra text):
{
  "avaliation": "<1–4 sentences>",
  "refined": "<positive prompt, comma-separated>",
  "negative": "<negative prompt, comma-separated; 6–8 items MAX + lighting, shadows, background>",
  "message": "<1–2 short sentences to the user>"
}
""".strip()

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I|re.M)
    return s.strip()

def chat_call(messages, *, json_mode=True, temperature=0.3, top_p=0.9, seed=None, timeout=180):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": temperature, "top_p": top_p} | ({"seed": seed} if seed is not None else {})
    }
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]

def start_chat_from_inputs(refined_init, negative_init, avaliation_txt):
    messages = [
        {"role": "system", "content": SYSTEM_CHAT},
        {"role": "user", "content":
            "START. Use the provided avaliation and current prompts; return ONLY the strict JSON.\n\n"
            f"- avaliation: {avaliation_txt}\n"
            f"- Current positive: {refined_init}\n"
            f"- Current negative: {negative_init}\n"
        }
    ]
    logger.info("[CHAT] STARTING CHAT")
    resp = chat_call(messages, json_mode=True)
    resp = _strip_fences(resp)
    try:
        data = json.loads(resp)
    except Exception:
        logger.warning(f"[CHAT] JSON parse failed at START. Using inputs. Raw: {resp}")
        data = {"avaliation": avaliation_txt, "refined": refined_init, "negative": negative_init, "message": "Loaded your initial evaluation and prompts. Tell me what to change; type 'ok' to finalize."}

    messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
    return messages, data

def continue_chat_with_feedback(messages, feedback, current_refined, current_negative, last_avaliation):
    logger.info(f"[CHAT] User feedback: {feedback}")
    logger.info(f"[CHAT] Messages: {messages}")
    # user accepted the current prompts > finish chat
    if user_accepted(feedback):
        data = {
            "avaliation": last_avaliation or "",
            "refined": current_refined,
            "negative": current_negative
        }
        # keep full history
        messages.append({"role": "user", "content": feedback})
        messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
        return messages, data, True  # <- finished=True

    # otherwise call teh model
    messages.append({"role": "user", "content":
        "FEEDBACK. Apply the client's requests by editing the prompts (add/remove/modify tokens). "
        "Return ONLY the strict JSON. Do not keep prompts unchanged if edits were requested.\n\n"
        f"- Client feedback: {feedback}\n"
        f"- Current avaliation: {last_avaliation or ''}\n"
        f"- Current positive: {current_refined}\n"
        f"- Current negative: {current_negative}\n"
    })
    logger.info("[CHAT] continue_chat_with_feedback")
    resp = chat_call(messages, json_mode=True)
    resp = _strip_fences(resp)
    try:
        data = json.loads(resp)
    except Exception:
        logger.warning(f"[CHAT] JSON parse failed at FEEDBACK. Raw: {resp}")
        data = {"avaliation": last_avaliation or "", "refined": current_refined, "negative": current_negative}

    messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
    return messages, data, False


# ------------------------------------------
OK_PAT = re.compile(r"\b(ok|okay|ok\.|ok!|accept|accepted|approve|approved|confirm|confirmed|finish|finalize|done|looks good|ship it|good to go)\b", re.I)

def user_accepted(text):
    return bool(text and OK_PAT.search(text.strip()))


def format_first_message(data):
    msg = data.get("message") or "Loaded your initial evaluation and prompts. Tell me what to change; type **ok** to finalize."
    eval_txt = data.get("avaliation") or ""
    return (
        f"{msg}\n\n"
        + (f"**Initial avaliation:**\n{eval_txt}\n\n" if eval_txt else "")
        + f"**Recommended positive prompt:**\n{data.get('refined','')}\n\n"
        + f"**Recommended negative prompt:**\n{data.get('negative','')}\n\n"
        + "Type **ok** to finalize."
    )

def format_turn_message(data, finished):
    msg = data.get("message")
    if finished:
        return (
            (f"{msg}\n\n" if msg else "")
            + "**Confirmed. Final prompts:**\n\n"
            + f"**Positive:** {data.get('refined','')}\n\n"
            + f"**Negative:** {data.get('negative','')}\n\n"
            + "Chat closed."
        )
    return (
        (f"{msg}\n\n" if msg else "")
        + f"**Positive (updated):**\n{data.get('refined','')}\n\n"
        + f"**Negative (updated):**\n{data.get('negative','')}\n\n"
        + "More changes? Type **ok** to finalize."
    )


# ------------------------------------------

if __name__ == "__main__":
    refined_init   = "park, big, trails, trees, central lake"
    negative_init  = "low detail, artifacts, duplicate trees, texture stretching, flat water, unrealistic reflections, incorrect scale, geometry holes, lighting, shadows, background"
    avaliation_txt = "Good multi-view consistency; foliage slightly repetitive; water surface looks flat; colors acceptable."

    # START
    msgs, start_out = start_chat_from_inputs(refined_init, negative_init, avaliation_txt)
    print("\n--- START ---")
    print(json.dumps(start_out, indent=2, ensure_ascii=False))

    feedbacks = [
        "i want more water, less trees, more green",
        "remove background and center the lake",
        "increase detail on water surface and avoid repetitive foliage",
        "reduce saturation overall, keep natural look",
    ]

    current = start_out
    for i, fb in enumerate(feedbacks, start=1):
        msgs, updated, _ = continue_chat_with_feedback(
            msgs,
            feedback=fb,
            current_refined=current["refined"],
            current_negative=current["negative"],
            last_avaliation=current["avaliation"]
        )
        logger.info(f"[USER FEEDBACK {i}] {fb}")
        print(f"\n--- AFTER FEEDBACK {i} ---")
        print(json.dumps(updated, indent=2, ensure_ascii=False))
        current = updated

    print("\n--- FINAL PROMPTS ---")
    print("Refined:", current["refined"])
    print("Negative:", current["negative"])
    print("Message:", current["message"])
