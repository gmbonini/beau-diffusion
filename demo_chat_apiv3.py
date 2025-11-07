import base64
import traceback
from typing import List, Tuple
import io
from ollama import check_views, prepare_prompts
from ollama_chat import start_chat_from_inputs, continue_chat_with_feedback, format_first_message, user_accepted, format_turn_message
import zipfile
import os, tempfile, glob, gc
import torch
from loguru import logger
import gradio as gr
from PIL import Image

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8081")

logger.add("gradio.log", rotation="1 MB")

def _sync_llm_prompts(cur_ref, cur_neg):
    return gr.update(value=cur_ref, visible=True), gr.update(value=cur_neg, visible=True)

def free_vram():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

# def _save_images_to_tmp(images):
#     tmpdir = tempfile.mkdtemp(prefix="views_")
#     for i, im in enumerate(images):
#         p = os.path.join(tmpdir, f"view_{i:02d}.png")
#         im.save(p)
#     return tmpdir

def _load_views_from_dir(views_dir):
    paths = sorted(glob.glob(os.path.join(views_dir, "*.png")))
    imgs = [Image.open(p).convert("RGB") for p in paths]
    return imgs, paths

# ------------- Chat ---------------
def chat_start_fn(refined, neg_txt, eval_txt):
    logger.info(f"[CHAT] starting chat with refined: {refined}, negative: {neg_txt}, eval: {eval_txt}")
    messages, data = start_chat_from_inputs(refined, neg_txt, eval_txt)
    first_bot = format_first_message(data)
    history = [("", first_bot)]
    return (
        history,
        messages,
        data["refined"],
        data["negative"],
        data["avaliation"],
        gr.update(value="", interactive=True),
        gr.update(visible=False, interactive=False)
    )

def chat_continue_fn(history, user_text, messages, cur_ref, cur_neg, cur_eval):
    if not user_text or not user_text.strip():
        return history, messages, cur_ref, cur_neg, gr.update(value="")
    messages, data, finished = continue_chat_with_feedback(messages, user_text, cur_ref, cur_neg, cur_eval)
    bot = format_turn_message(data, finished)
    return (
        history + [(user_text, bot)],
        messages,
        data["refined"],
        data["negative"],
        gr.update(value="", interactive=not finished),
        gr.update(visible=finished, interactive=finished)
    )

 # ------------ MV-Adapter ------------
def llm_prompt_processing(prompt):
    logger.info(f"[MV-ADAPTER] generating images. prompt to be refined: {prompt}")
    result = prepare_prompts(prompt)
    ref_prompt = result['refined']
    neg_prompt = result['negative'] or "nothing to change"
    return (
        gr.update(value=ref_prompt, visible=True),
        gr.update(value=neg_prompt, visible=True),
        ref_prompt,
        neg_prompt
    )

def _api_t2mv_generate(refined_prompt, negative_prompt, randomize, inference_steps=40):
    logger.info(f"[MV-ADAPTER] Calling API with {inference_steps} steps...")
    
    resp = requests.post(
        f"{API_URL}/t2mv/generate",
        params={
            "ref_prompt": refined_prompt, 
            "neg_prompt": negative_prompt, 
            "randomize": randomize,
            "inference_steps": inference_steps
        },
        timeout=600,
    )
    resp.raise_for_status()
    logger.info(f"[MV-ADAPTER] API call successful, processing response...")    
    views_dir = tempfile.mkdtemp(prefix="views_")
        
    with zipfile.ZipFile(io.BytesIO(resp.content), 'r') as zipf:
        zipf.extractall(views_dir)
    
    logger.info(f"[MV-ADAPTER] Extracted ZIP to {views_dir}")
    return views_dir


def generate_images(refined_prompt, negative_prompt, randomize=False, inference_steps=40):
    logger.info(f"[MV-ADAPTER] Starting generation with {inference_steps} inference steps...")
    try:
        views_dir = _api_t2mv_generate(refined_prompt, negative_prompt, randomize, inference_steps)
        paths = sorted(glob.glob(os.path.join(views_dir, "*.png")))
        logger.info(f"[MV-ADAPTER] Got {len(paths)} images in {views_dir}")
        
        gallery_paths = paths
                
        llm_avaliation_update = gr.update(value="", visible=False)
        llm_eval_neg_prompt_update = gr.update(value="", visible=False)

        return (
            gallery_paths,                 # gallery
            views_dir,                     # state_views_dir
            gr.update(visible=True),       # run_trellis_btn (tornar visível)
            refined_prompt,                # state_last_prompt
            llm_avaliation_update,         # llm_avaliation
            llm_eval_neg_prompt_update     # llm_eval_neg_prompt
        )
    except Exception as e:        
        logger.exception(f"[MV-ADAPTER] Error generating images: {e}")        
        return (
            [],                            # gallery vazio
            "",                            # state_views_dir vazio
            gr.update(visible=False),      # run_trellis_btn escondido
            refined_prompt or "",          # state_last_prompt (tentar não perder o prompt)
            gr.update(value=f"Error: {e}", visible=True),  # llm_avaliation mostra erro
            gr.update(value="", visible=False)             # llm_eval_neg_prompt
        )

def check_views_quality(views_dir, prompt):
    result = check_views(views_dir, prompt)
    llm_avaliation = result['avaliation']
    llm_neg_prompt = result['negative_prompt']

    logger.info(f"[MV-ADAPTER] views quality check: {result}")
    return llm_avaliation, llm_neg_prompt

 # ------------ TRELLIS ------------

def _b64_to_file(b64_str: str, path: str):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))

def generate_3d(views_dir):
    images_list, paths = _load_views_from_dir(views_dir)
    assert len(images_list) >= 2, "need at least 2 views to generate 3D"

    files = [("files", (os.path.basename(p), open(p, "rb"), "image/png")) for p in paths]

    r = requests.post(f"{API_URL}/mv2m/generate", files=files, timeout=900)
    r.raise_for_status()
    data = r.json()

    glb_path = os.path.join(views_dir, data["filenames"]["glb"])
    ply_path = os.path.join(views_dir, data["filenames"]["ply"])
    mp4_path = os.path.join(views_dir, data["filenames"]["mp4"])

    _b64_to_file(data["glb_b64"], glb_path)
    _b64_to_file(data["ply_b64"], ply_path)
    _b64_to_file(data["mp4_b64"], mp4_path)

    frame_path = data.get("frame_path")

    if frame_path:
        logger.info(f"[TRELLIS/API-path] Frame path received: {frame_path}")
    else:
        logger.warning("[TRELLIS/API-path] No frame path returned from API.")

    logger.info(f"[TRELLIS/API-b64] assets: {glb_path}, {ply_path}, {mp4_path}")

    return (
        mp4_path,
        gr.update(value=glb_path, visible=True),
        gr.update(value=ply_path, visible=True),
        frame_path
    )

 # ------------ UI helpers ------------
def _lock_all_buttons():
    return (
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )

def _unlock_after_generate():
    return (
        gr.update(interactive=True),
        gr.update(visible=True, interactive=True),
        gr.update(visible=True, interactive=True),
        gr.update(visible=True, interactive=True),
        gr.update(visible=True, interactive=True),
    )

def _unlock_after_trellis():
    return (
        gr.update(interactive=True),
        gr.update(visible=True, interactive=True),
        gr.update(visible=True, interactive=True),
        gr.update(visible=True, interactive=True),
        gr.update(visible=True, interactive=True),
    )

def show_negative_feedback_input():
    """Show the negative feedback text input and send button"""
    return gr.update(visible=True, value=""), gr.update(visible=True)

def send_review(
    feedback_type: str,
    original_prompt: str,
    refined: str,
    chat_history: List[Tuple[str, str]] = None,
    negative_prompt: str = "",
    video_frame_path: str = None,
    multiview_dir: str = None,
    step: str = None,
    user_feedback_text: str = ""
):
    is_positive_val = 1 if feedback_type.lower() in ["like", "thumbs up", "👍"] else 0

    chat_text = ""
    if chat_history:
        chat_lines = []        
        for msg in chat_history:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'user':
                chat_lines.append(f"User: {content}")
            elif role == 'assistant':
                chat_lines.append(f"Bot: {content}")            
        
        chat_text = "\n".join(chat_lines)

    data = {
        "is_positive": str(is_positive_val),
        "original_prompt": original_prompt,
        "refined": refined,
        "chat": chat_text,
        "negative_feedback": user_feedback_text if not is_positive_val else "",
        "negative_prompt": negative_prompt,
        "video_frame_path": video_frame_path if video_frame_path else "",
        "step" : step
    }

    files_list = []
    file_handles_to_close = []

    if multiview_dir and os.path.exists(multiview_dir):
        multiview_images = sorted(glob.glob(os.path.join(multiview_dir, "*.png")))

        for file_path in multiview_images:
            file_name = os.path.basename(file_path)
            try:
                file_handle = open(file_path, "rb")
                file_handles_to_close.append(file_handle)

                files_list.append(
                    ("multiview_files", (file_name, file_handle, "image/png"))
                )
            except IOError as e:
                logger.error(f"Não foi possível abrir o arquivo multiview {file_path}: {e}")
                continue

    files_dict_for_request = files_list if files_list else None

    try:
        response = requests.post(f"{API_URL}/feedback/save", data=data, files=files_dict_for_request, timeout=60)
        response.raise_for_status()
        message = "Feedback sent!"
        logger.info(f"[FEEDBACK] Success: {message}")
    except Exception as e:
        logger.error(f"[FEEDBACK] Error: {e}")
        message = f"Error sending feedback! {str(e)}"
    finally:
        for file_handle in file_handles_to_close:
            file_handle.close()

    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True, value=message),
        gr.update(visible=False, value=""),  # Hide and clear feedback input
        gr.update(visible=False),  # Hide send button
    )

with gr.Blocks() as demo:
    gr.Markdown("llm chat demo")

    state_inference_steps = gr.State(value=40)
    state_views_dir = gr.State(value="")
    state_last_prompt = gr.State(value="")
    state_neg_prompt = gr.State(value="")
    state_chat_msgs = gr.State(value=[])
    state_full_chat = gr.State(value=[])
    state_chat_msgs_before = gr.State(value=[])
    state_eval = gr.State(value="")
    state_frame_path = gr.State(value=None)
    remake_state = gr.State(0)
    
    prompt = gr.Textbox(label="Prompt")
    run_btn = gr.Button("Generate")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=320):
            with gr.Accordion("LLM prompts", open=True):
                ref_prompt = gr.Textbox(
                    label="Refined positive prompt",
                    interactive=False,
                    visible=False,
                    show_copy_button=True,
                    lines=2,
                )
                neg_prompt = gr.Textbox(
                    label="Negative prompt",
                    interactive=False,
                    visible=False,
                    show_copy_button=True,
                    lines=2,
                )

        with gr.Column(scale=2, min_width=440):
            gallery = gr.Gallery(columns=3, rows=2, height=520)
            remake_btn = gr.Button("Remake images (same prompt but different seed and more steps)", visible=False)
            with gr.Row():
                image_like_btn = gr.Button("👍", visible=False)
                image_dislike_btn = gr.Button("👎", visible=False)
            
            # Negative feedback text input for images
            with gr.Row(equal_height=True):
                image_feedback_text = gr.Textbox(
                    label="What was the problem? (optional)",
                    placeholder="Describe what was wrong with the images...",
                    visible=False,
                    lines=2,
                    scale=4,
                )
                image_feedback_send_btn = gr.Button(
                    "Send",
                    visible=False,
                    size="sm",
                    scale=1,                    
                )
            
            image_review_status = gr.Textbox(
                label="Feedback Status",
                value="",
                visible=False,
                interactive=False,
            )

        with gr.Column(scale=1, min_width=360):
            with gr.Accordion("Adjustments Chat", open=True):
                chatbot = gr.Chatbot(label="Conversation", value=[], height=520)
                user_msg = gr.Textbox(
                    label="Your message",
                    placeholder="remove background, more water, center the lake... (type 'ok' to finalize)",
                    lines=1,
                )
                
            apply_chat_btn = gr.Button("Regenerate images with chat prompts", visible=False)

    with gr.Accordion("LLM Image Avaliation", open=False):
        llm_avaliation = gr.Textbox(
            label="LLM Image Avaliation",
            interactive=False,
            visible=False,
            show_copy_button=True,
            lines=2
        )
        llm_eval_neg_prompt = gr.Textbox(
            label="LLM Prompt Negative using image avaliation",
            interactive=False,
            visible=False,
            show_copy_button=True,
            lines=2
        )

    run_trellis_btn = gr.Button("Generate 3D Mesh", visible=False)
    video = gr.Video(label="Mesh video preview", interactive=False)

    with gr.Row():
        video_like_btn = gr.Button("👍", visible=False)
        video_dislike_btn = gr.Button("👎", visible=False)

    # Negative feedback text input for video
    with gr.Row():
        video_feedback_text = gr.Textbox(
            label="What was the problem? (optional)",
            placeholder="Describe what was wrong with the 3D mesh...",
            visible=False,
            lines=2,
            scale=4
        )
        video_feedback_send_btn = gr.Button("Send", visible=False, scale=1, size="sm")
    
    review_status = gr.Textbox(
        label="Feedback Status",
        value="Review sent. Thank you!",
        visible=False,
        interactive=False,
    )

    with gr.Row():
        download_glb = gr.DownloadButton(
            label="Download GLB",
            value=None,
            visible=False,
            size="sm"
        )
        download_ply = gr.DownloadButton(
            label="Download PLY",
            value=None,
            visible=False,
            size="sm"
        )

    # GENERATE
    run_btn.click(
        fn=_lock_all_buttons, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn, apply_chat_btn, image_like_btn, image_dislike_btn]
    ).then(
    fn=lambda: (
        gr.update(value="", visible=False), gr.update(visible=False), 
        gr.update(visible=False), [], [], [], 0,
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        40,
        gr.update(value="", visible=False),  # image_feedback_text
        gr.update(visible=False),  # image_feedback_send_btn
        gr.update(value="", visible=False),  # video_feedback_text
        gr.update(visible=False),  # video_feedback_send_btn
    ),
    inputs=None,
    outputs=[
        image_review_status, image_like_btn, image_dislike_btn,
        chatbot, state_chat_msgs, state_full_chat, remake_state,
        llm_avaliation, llm_eval_neg_prompt,
        state_inference_steps,
        image_feedback_text,
        image_feedback_send_btn,
        video_feedback_text,
        video_feedback_send_btn
    ]     
    ).then(
        fn=llm_prompt_processing,
        inputs=prompt,
        outputs=[ref_prompt, neg_prompt, state_last_prompt, state_neg_prompt],
    ).then(
        fn=generate_images,
        inputs=[state_last_prompt, state_neg_prompt, gr.State(False), state_inference_steps],
        outputs=[gallery, state_views_dir, run_trellis_btn, state_last_prompt, llm_avaliation, llm_eval_neg_prompt],
        queue=True
    ).then(
        fn=check_views_quality,
        inputs=[state_views_dir, state_last_prompt],
        outputs=[llm_avaliation, llm_eval_neg_prompt]
    ).then(
        fn=chat_start_fn,
        inputs=[state_last_prompt, state_neg_prompt, llm_avaliation],
        outputs=[chatbot, state_chat_msgs, state_last_prompt, state_neg_prompt, state_eval, user_msg, apply_chat_btn]
    ).then(
        fn=lambda new_chat_msgs: new_chat_msgs,
        inputs=[state_chat_msgs],
        outputs=[state_full_chat]
    ).then(
        fn=_unlock_after_generate, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn, image_like_btn, image_dislike_btn]
    )
    
    state_views_dir.change(
        fn=lambda: [],
        inputs=None,
        outputs=[state_full_chat]
    ).then(
        fn=chat_start_fn,
        inputs=[state_last_prompt, state_neg_prompt, llm_avaliation],
        outputs=[chatbot, state_chat_msgs, state_last_prompt, state_neg_prompt, state_eval, user_msg, apply_chat_btn]
    ).then(
        fn=lambda new_chat_msgs: new_chat_msgs,
        inputs=[state_chat_msgs],
        outputs=[state_full_chat]
    ).then(
        fn=_unlock_after_generate, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn, image_like_btn, image_dislike_btn]
    )
    
    user_msg.submit(
        fn=lambda current_msgs: current_msgs,
        inputs=[state_chat_msgs],
        outputs=[state_chat_msgs_before]
    ).then(
        fn=chat_continue_fn,
        inputs=[chatbot, user_msg, state_chat_msgs, state_last_prompt, state_neg_prompt, state_eval],
        outputs=[chatbot, state_chat_msgs, state_last_prompt, state_neg_prompt, user_msg, apply_chat_btn],
        queue=True
    ).then(
        fn=lambda full_hist, new_hist, old_hist: full_hist + new_hist[len(old_hist):],
        inputs=[state_full_chat, state_chat_msgs, state_chat_msgs_before],
        outputs=[state_full_chat]
    )

    # remake images using chat prompt
    apply_chat_btn.click(
        fn=lambda: 1,
        inputs=None,
        outputs=remake_state
    ).then(
        fn=_lock_all_buttons, 
        inputs=None, 
        outputs=[run_btn, remake_btn, run_trellis_btn, apply_chat_btn, image_like_btn, image_dislike_btn]
    ).then(
        fn=_sync_llm_prompts,
        inputs=[state_last_prompt, state_neg_prompt],
        outputs=[ref_prompt, neg_prompt]
    ).then(
        fn=generate_images,
        inputs=[state_last_prompt, state_neg_prompt, gr.State(False), state_inference_steps],
        outputs=[gallery, state_views_dir, run_trellis_btn, state_last_prompt, llm_avaliation, llm_eval_neg_prompt],
        queue=True
    )
    
    state_views_dir.change(
        fn=chat_start_fn,
        inputs=[state_last_prompt, state_neg_prompt, llm_avaliation],
        outputs=[chatbot, state_chat_msgs, state_last_prompt, state_neg_prompt, state_eval, user_msg, apply_chat_btn]
    ).then(
        fn=lambda full_hist, new_start_msg: full_hist + new_start_msg,
        inputs=[state_full_chat, state_chat_msgs],
        outputs=[state_full_chat]
    ).then(
        fn=lambda: gr.update(value="", visible=False),
        inputs=None,
        outputs=image_review_status
    ).then(
        fn=_unlock_after_generate, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn, image_like_btn, image_dislike_btn]
    )

    # remake
    remake_btn.click(
        fn=lambda remake, current_steps: (1, current_steps + 5),
        inputs=[remake_state, state_inference_steps],
        outputs=[remake_state, state_inference_steps],
    ).then(
        fn=_lock_all_buttons,
        inputs=None,
        outputs=[run_btn, remake_btn, run_trellis_btn, apply_chat_btn, image_like_btn, image_dislike_btn]
    ).then(        
        fn=lambda: gr.update(value="", visible=False),
        inputs=None,
        outputs=image_review_status    
    ).then(
        fn=generate_images,
        inputs=[state_last_prompt, state_neg_prompt, gr.State(True), state_inference_steps],
        outputs=[gallery, state_views_dir, run_trellis_btn, state_last_prompt, llm_avaliation, llm_eval_neg_prompt]
    ).then(        
        fn=chat_start_fn, 
        inputs=[state_last_prompt, state_neg_prompt, llm_avaliation],
        outputs=[chatbot, state_chat_msgs, state_last_prompt, state_neg_prompt, state_eval, user_msg, apply_chat_btn]
    ).then(        
        fn=lambda full_hist, new_start_msg: full_hist + new_start_msg, 
        inputs=[state_full_chat, state_chat_msgs],
        outputs=[state_full_chat]
    ).then(
        fn=lambda: gr.update(value="", visible=False),
        inputs=None,
        outputs=image_review_status
    ).then(
        fn=_unlock_after_generate,
        inputs=None,
        outputs=[run_btn, remake_btn, run_trellis_btn, image_like_btn, image_dislike_btn]
    )

    # trellis
    run_trellis_btn.click(
        fn=_lock_all_buttons, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn, apply_chat_btn, image_like_btn, image_dislike_btn]
    ).then(
        fn=generate_3d, inputs=state_views_dir, outputs=[video, download_glb, download_ply, state_frame_path]
    ).then(
        fn=_unlock_after_trellis, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn, video_like_btn, video_dislike_btn]
    )

    # feedback helpers
    def _get_feedback_click(feedback_type, step=None, dynamic_step=False):
        def inner(orig, refined, chat, neg, video_path, views_dir, current_remake_value, feedback_text):
            current_step = (                
                "REGENERATE" if dynamic_step and current_remake_value == 1
                else ("IMAGE" if dynamic_step else step)
            )

            return send_review(
                feedback_type=feedback_type,
                original_prompt=orig,
                refined=refined,
                chat_history=chat,
                negative_prompt=neg,
                video_frame_path=video_path,
                multiview_dir=views_dir,
                step=current_step,
                user_feedback_text=feedback_text
            )
        return inner

    dummy_state = gr.State()
    
    video_like_btn.click(
        fn=_get_feedback_click("like", step="VIDEO"),              
        inputs=[prompt, state_last_prompt, state_full_chat, state_neg_prompt, state_frame_path, state_views_dir, dummy_state, gr.State("")],
        outputs=[video_like_btn, video_dislike_btn, review_status, video_feedback_text, video_feedback_send_btn],
        queue=False
    )
    
    video_dislike_btn.click(
        fn=show_negative_feedback_input,
        inputs=None,
        outputs=[video_feedback_text, video_feedback_send_btn],
        queue=False
    )
    
    video_feedback_send_btn.click(
        fn=_get_feedback_click("dislike", step="VIDEO"),              
        inputs=[prompt, state_last_prompt, state_full_chat, state_neg_prompt, state_frame_path, state_views_dir, dummy_state, video_feedback_text],
        outputs=[video_like_btn, video_dislike_btn, review_status, video_feedback_text, video_feedback_send_btn],
        queue=False
    )
    
    image_like_btn.click(
        fn=_get_feedback_click("like", dynamic_step=True),        
        inputs=[prompt, state_last_prompt, state_full_chat, state_neg_prompt, dummy_state, state_views_dir, remake_state, gr.State("")],
        outputs=[image_like_btn, image_dislike_btn, image_review_status, image_feedback_text, image_feedback_send_btn],
        queue=False
    )
    
    image_dislike_btn.click(
        fn=show_negative_feedback_input,
        inputs=None,
        outputs=[image_feedback_text, image_feedback_send_btn],
        queue=False
    )
    
    image_feedback_send_btn.click(
        fn=_get_feedback_click("dislike", dynamic_step=True),        
        inputs=[prompt, state_last_prompt, state_full_chat, state_neg_prompt, dummy_state, state_views_dir, remake_state, image_feedback_text],
        outputs=[image_like_btn, image_dislike_btn, image_review_status, image_feedback_text, image_feedback_send_btn],
        queue=False
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)