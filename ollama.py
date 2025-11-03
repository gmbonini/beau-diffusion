import base64
import glob
import requests
import os
import json
from loguru import logger

logger.add("ollama.log", rotation="10 MB")

# nsfw checker
# mesh checker

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11200")
MODEL_TEXT = os.getenv("TEXT_MODEL", "qwen2.5:7b")
MODEL_VISION = os.getenv("VISION_MODEL", "qwen2.5vl:7b")

# testar com modificadores de qualidade

def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def start_ollama(model=None, keep="2h"):
    """Inicia o modelo especificado ou ambos se None"""
    models_to_start = [model] if model else [MODEL_TEXT, MODEL_VISION]
    
    for m in models_to_start:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": m, "prompt": "ok", "stream": False, "keep_alive": keep},
            timeout=(3, 15), # 3 to connect 15 to receive response
        )
        r.raise_for_status()
    return True

def is_ollama_loaded(model=None):
    """Verifica se um modelo específico está carregado, ou ambos se None"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        data = resp.json()
        loaded_models = {m.get("model") for m in data.get("models", []) if m.get("loaded")}
        
        if model:
            return model in loaded_models
        
        return MODEL_TEXT in loaded_models
    except Exception:
        return False
    
def prepare_prompts(user_prompt: str) -> dict:
    """Usa modelo de TEXTO para processar prompts"""
    prompt_text = f"""
        You are a prompt engineer for text-to-3D/multiview generation.
        YOU HAVE TWO TASKS.

        [TASK 1: REFINED POSITIVE PROMPT]
            Goal: rewrite a messy input into a clean, structured description that keeps multi-word phrases intact.
            Rules:
            - Start with the object name (singular, lowercase). Example: "orange cat sitting on a table" > start with "cat".
            - Ignore background or environment details. Focus on the main object. Example: "dragon flying in the sky" > "dragon", "flying".
            - Use concise, visual attributes.
            - Do not ignore a action if it is important to the object. Example: "dragon flying" > "dragon", "flying"; "cat sitting" > "cat", "sitting". ; cat drinking white milk > cat, drinking white milk
            - Keep compound phrases together (examples: "lake in the middle", "red door on the left", "snow-capped mountain").
            - Preserve spatial relations ("at the center", "on the left/right", "surrounded by").
            - Do NOT split a phrase into separate tokens like "lake, middle". Keep it as "central lake" or "lake in the middle".
            - Prefer synonyms that compress relations: "lake in the middle" to "central lake".
            - Output ONLY the refined prompt.
            - Ever add "full body" if the prompt is about an animal or human. You don't need to add "full body" for objects, landscapes, scenes and if the prompt already implies a specific part of the object (like "head of a cat", "face of a person", "top of a mountain").
            - No extra text.

            Examples:
            Input: "a cute orange cat"
            Output: "cat, cute, orange, detailed fur, full body"
            Input: "a house with a red door on the left and a big tree behind"
            Output: "house, red door on the left, big tree behind"

            Input: "a city square with a fountain in the center and benches around"
            Output: "city square, central fountain, benches around"

            Input: "a big park with trails and trees and a big lake in the middle"
            Output: "park, big, trails, trees, central lake"

        [TASK 2: NEGATIVE PROMPT]
            Goal: given a positive prompt, generate a concise negative prompt that lists unwanted attributes to avoid in the generated output. Follow strictly these instructions:
            Rules:
            - Use concise, visual attributes.
            - 6-8 items max. Do NOT exceed 8 items. NEVER give a list longer than 8 items.
            - Focus on common issues in 3D generation: "blurry", "low detail", "bad anatomy"; but related to the object.
            - Be context-aware. Also focus on the prompt object and give relevant negative attributes related to it. Example: for "cat", include "extra limbs", "extra tail", "deformed face".
                * Humans/animals: anatomy terms allowed (example: extra limbs, deformed face).
                * Landscapes/scenes/objects: DO NOT use anatomy words or "incorrect <object>" phrases. Prefer concrete artifact terms (topology, textures, composition).
            - Prefer general artifact terms + a few scene-specific constraints. No redundancy.
            - Avoid overly specific or rare terms.
            - Avoid vague terms like "weird", "strange", "bad".
            - EVER add "lighting", "shadows", "background". We just want to focus on the main object. This can exceed the 8 item limit. Every prompt must have these three terms. It's not incorrent shadow or lighting, just "shadows" and "lighting".
            - Avois insert "incorrent <object>" phrases. Specific anatomy terms are allowed only for humans/animals/objects. Cite a problem that makes sense for the object. Example: don't say "incorrect car" for a car prompt. Instead, say "deformed wheels" or "bad car proportions".
            - Output ONLY the negative prompt and on one line with commas separating attributes.
            - No extra text.

            Input: park, big, trails, trees, central lake
            Output: low detail, artifacts, duplicate trees, intersecting trails, overcrowded foliage, floating objects, geometry holes, texture stretching, tiling textures, flat water, blocky water surface, unrealistic water reflections, incorrect scale

        RETURN FORMAT (MANDATORY):
        Return ONLY this JSON (no extra text):
        {{
        "refined": "<refined positive prompt from TASK 1>",
        "negative": "<negative prompt from TASK 2. 6-8 ITEMS MAX (one line)>"
        }}

        Input: {user_prompt}
        """.strip()

    logger.info(f"[OLLAMA] prepare_prompts: preparing prompts for input: {user_prompt} (using {MODEL_TEXT})")
    
    payload = {
        "model": MODEL_TEXT,
        "prompt": prompt_text,
        "max_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9,
        "stream": False,
        "keep_alive": "1h",
        "format": "json"
    }

    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=(10, 120))
    resp.raise_for_status()
    raw = resp.json()["response"].strip()

    data = json.loads(raw)
    print(data)
    logger.info(f"[MV-ADAPTER] prepare_prompts result: {data}")

    return data


# def refine_prompt(user_prompt):
#     """Usa modelo de TEXTO para refinar prompts"""
#     payload ={
#         "model": MODEL_TEXT,
#         "prompt": f"""
#             You are a prompt engineer for text-to-3D/multiview generation.
#             Goal: rewrite a messy input into a clean, structured description that keeps multi-word phrases intact.
#             Rules:
#             - Start with the object name (singular, lowercase).
#             - Use concise, visual attributes.
#             - Keep compound phrases together (examples: "lake in the middle", "red door on the left", "snow-capped mountain").
#             - Preserve spatial relations ("at the center", "on the left/right", "surrounded by").
#             - Do NOT split a phrase into separate tokens like "lake, middle". Keep it as "central lake" or "lake in the middle".
#             - Prefer synonyms that compress relations: "lake in the middle" to "central lake".
#             - Output ONLY the refined prompt.
#             - No extra text.

#             Examples:
#             Input: "a cute orange cat"
#             Output: "cat, cute, orange, detailed fur"
# ,
#             Input: "a house with a red door on the left and a big tree behind"
#             Output: "house, red door on the left, big tree behind"

#             Input: "a city square with a fountain in the center and benches around"
#             Output: "city square, central fountain, benches around"

#             Input: "a big park with trails and trees and a big lake in the middle"
#             Output: "park, big, trails, trees, central lake"

#             Refine this prompt: {user_prompt}
#             """,
#         "max_tokens": 256,
#         "temperature": 0.7,
#         "top_p": 0.9,
#         "n": 1,
#         "stream": False,
#         "keep_alive": "1h"
        
#     }
#     logger.info(f"[MV-ADAPTER] refine prompt loading. POST /api/generate to {OLLAMA_URL} (using {MODEL_TEXT})")
#     resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
#     resp.raise_for_status()
#     refined_prompt = resp.json()['response']
#     logger.info(f"[MV-ADAPTER] refined prompt result: {refined_prompt}")
#     return refined_prompt

# def negative_prompt(user_prompt):
#     """Usa modelo de TEXTO para gerar negative prompts"""
#     payload ={
#         "model": MODEL_TEXT,
#         "prompt": f"""
#             You are a prompt engineer for text-to-3D/multiview generation.
#             Goal: given a positive prompt, generate a concise negative prompt that lists unwanted attributes to avoid in the generated output.
#             Rules:
#             - Use concise, visual attributes.
#             - 6-8 items max. Do NOT exceed 8 items. NEVER give a list longer than 8 items. The negative prompt cannot pass 8 items.
#             - Focus on common issues in 3D generation: "blurry", "low detail", "bad anatomy"; but related to the object.
#             - Be context-aware. Also focus on the prompt object and give relevant negative attributes related to it. Example: for "cat", include "extra limbs", "extra tail", "deformed face".
#                 * Humans/animals: anatomy terms allowed (example: extra limbs, deformed face).
#                 * Landscapes/scenes/objects: DO NOT use anatomy words or "incorrect <object>" phrases. Prefer concrete artifact terms (topology, textures, composition).
#             - Prefer general artifact terms + a few scene-specific constraints. No redundancy.
#             - Avoid overly specific or rare terms.
#             - Avoid vague terms like "weird", "strange", "bad".
#             - Avois insert "incorrent <object>" phrases. Specific anatomy terms are allowed only for humans/animals/objects. Cite a problem that makes sense for the object. Example: don't say "incorrect car" for a car prompt. Instead, say "deformed wheels" or "bad car proportions".
#             - EVER add "lighting", "shadows", "background". We just want to focus on the main object. This can exceed the 8 item limit.
#             - Output ONLY the negative prompt and on one line with commas separating attributes.
#             - No extra text.

#             Input: park, big, trails, trees, central lake
#             Output: low detail, artifacts, duplicate trees, intersecting trails, overcrowded foliage, floating objects, geometry holes, texture stretching, tiling textures, flat water, blocky water surface, unrealistic water reflections, incorrect scale

#             Generate the negative prompt: {user_prompt}
#             """,
#         "max_tokens": 256,
#         "temperature": 0.7,
#         "top_p": 0.9,
#         "n": 1,
#         "stream": False,
#         "keep_alive": "1h"
#     }

#     logger.info(f"[MV-ADAPTER] negative prompt loading. POST /api/generate to {OLLAMA_URL} (using {MODEL_TEXT})")
#     resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
#     resp.raise_for_status()
#     refined_prompt = resp.json()['response']
#     logger.info(f"[MV-ADAPTER] negative prompt result: {refined_prompt}")
#     return refined_prompt

def check_views(dir_path, user_prompt):
    """Usa modelo de VISÃO para análise de imagens"""
    pngs = sorted(glob.glob(os.path.join(dir_path, "*.png")))

    prompt_text = f"""
        You are an image quality reviewer for 3D multiview generation.

        Given the target prompt and the attached images, analyze what you see.
        You need to evaluate:
        1) How well do the images match the prompt? (accuracy to the prompt)
        2) Are the images consistent with each other? (consistency across the multi views. Remember, it's a SINGLE object from DIFFERENT angles)
        3) Identify any visible issues/artifacts in the images. (image quality)

        You can be critical but not too much in your evaluation. If the images have minor issues but overall look good, say so. If they have major problems, point them out clearly. Only give constructive feedback.
        If the overall quality is good, you don't need to list every small issue and the negative prompt can be "none".
        The bigger problems are: geometry holes, extra limbs (if the prompt doesn't call for it), deformed face, artifacts, lighting, shadows, background.
        The images need to be realistic. For example: if the input is "a cat", the images should look like a real cat from different angles. If the cat has extra limbs, deformed face, a extra tail, say so.
        Return your answer strictly as a JSON object with two fields:

        {{
        "avaliation": "<describe if the images match the prompt, consistency, visible issues>",
        "negative_prompt": "<comma-separated short negative prompt with 6–8 concise unwanted attributes to avoid; if it is good, write 'none'>"
        }}

        ONLY AVALIATION and NEGATIVE_PROMPT as keys.
        Positive prompt: {user_prompt}

        Output format example:
        {{
        [avaliation]: the cat have a extra tail and a deformed body. The images are not consistent and some views have a extra tail. the face has low detail. the colors are unrealistic and unmatched.
        [negative_prompt]: blurry, low detail, artifacts, geometry holes, extra tail.
        }}
        Return ONLY the JSON, no extra text.
        """.strip()


    payload = {
        "model": MODEL_VISION,  # USA MODELO DE VISÃO
        "prompt": prompt_text,
        "images": [_b64(p) for p in pngs],
        "max_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9,
        "stream": False,
        "keep_alive": "1h",
        "format": "json"
    }
    logger.info(f"[MV-ADAPTER] check_views loading (using {MODEL_VISION})")
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    resp.raise_for_status()
    raw = resp.json()["response"].strip()

    data = json.loads(raw)
    logger.info(f"[MV-ADAPTER] check_views result: {data}")
    return data


if __name__ == "__main__":
    # print(refine_prompt("a big park with trails and trees and big lake in the middle"))
    # print(negative_prompt("park, big, trails, trees, central lake"))
    # describe_pngs_in_dir("/tmp/views_ibp6tnyi/")
    # result = check_views("/tmp/views_ibp6tnyi/", "a big fat cat with big tail")
    # print(result)
    # print(result['avaliation'])
    prompts = prepare_prompts("a big park with trails and trees and big lake in the middle")
    print(prompts['refined'])
    print(prompts['negative'])


# old prompts:
"""
    You are a prompt engineer for text-to-3D mesh generation.
    Your task is to rewrite a user vague or messy input into a clean, structured positive prompt that only describes the object itself.
    You need to: always start with the object name, separate descriptors with commas.
    Only output the refined prompt, do not include any explanations or additional text.
    Here are some examples:
    User input: "a cute orange cat"
    Refined prompt: "cat, cute, orange, detailed fur"
    Refine this prompt: "{user_prompt}
"""


# ----------------- TESTING
def check_image(image_path):
    """Usa modelo de VISÃO para análise de imagem única"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": MODEL_VISION,  # USA MODELO DE VISÃO
        "prompt": "Describe what you see in the image in one concise sentence.",
        "images": [b64],
        "max_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
        "keep_alive": "1h",
    }
    logger.info(f"[TESTING] check_image loading (using {MODEL_VISION})")
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()

def describe_pngs_in_dir(dir_path):
    """Usa modelo de VISÃO para descrever múltiplas imagens"""
    pngs = sorted(glob.glob(os.path.join(dir_path, "*.png")))
    results = []
    for p in pngs:
        desc = check_image(p)
        print(f"{os.path.basename(p)}: {desc}")
        results.append((p, desc))
    return results