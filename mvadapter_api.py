from mvadapter_t2mv import MVAdapterT2MV
from trellis_mv2m import TrellisMV2M
from ollama import refine_prompt, negative_prompt, check_views, prepare_prompts
from ollama_chat import start_chat_from_inputs, continue_chat_with_feedback, format_first_message, user_accepted, format_turn_message

import os, tempfile, glob, gc
import torch
from loguru import logger
import gradio as gr
from PIL import Image

# LOCK REGENERATE BUTTON LLM
# ATT LLM PROMPT WITH CHAT PROMPT


logger.add("gradio.log", rotation="1 MB")

# att llm prompt
def _sync_llm_prompts(cur_ref, cur_neg):
    return gr.update(value=cur_ref, visible=True), gr.update(value=cur_neg, visible=True)


def free_vram():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

def _save_images_to_tmp(images):
    # save pil image list to a temp dir and return the dir
    tmpdir = tempfile.mkdtemp(prefix="views_")
    for i, im in enumerate(images):
        p = os.path.join(tmpdir, f"view_{i:02d}.png")
        im.save(p)
    return tmpdir

def _load_views_from_dir(views_dir):
    # load and order views by name
    paths = sorted(glob.glob(os.path.join(views_dir, "*.png")))
    imgs = [Image.open(p).convert("RGB") for p in paths]
    return imgs, paths

 # ------------ MV-Adapter ------------
pipe_mv, adapters = None, []
def load_pipe_mvadapter():
    global pipe_mv, adapters
    pipe_mv, adapters = MVAdapterT2MV.prepare_pipeline(
        base_model="Lykon/dreamshaper-xl-1-0", # Lykon/dreamshaper-xl-1-0 # stabilityai/stable-diffusion-xl-base-1.0
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        unet_model=None,
        lora_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler="ddpm",
        num_views=6,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    logger.info("[MV-ADAPTER] Pipeline loaded")

def llm_prompt_processing(prompt):
    # ref_prompt = refine_prompt(prompt)
    # neg_prompt = negative_prompt(ref_prompt)
    logger.info(f"[MV-ADAPTER] generating images. prompt to be refined: {prompt}")
    result = prepare_prompts(prompt)
    ref_prompt = result['refined']
    neg_prompt = result['negative'] or "nothing to change"
    # logger.info(f"[MV-ADAPTER] refined prompt: {ref_prompt}")
    # logger.info(f"[MV-ADAPTER] negative prompt: {neg_prompt}")
    return (
        gr.update(value=ref_prompt, visible=True), # ref_prompt
        gr.update(value=neg_prompt, visible=True), # neg_prompt
        ref_prompt, # state_last_prompt
        neg_prompt # state_neg_prompt
    )

def generate_images(refined_prompt, negative_prompt, randomize=False):
    seed = -1 if randomize else 42

    logger.info(f"[MV-ADAPTER] generating images. refined prompt: {refined_prompt}")
    logger.info(f"[MV-ADAPTER] negative prompt: {negative_prompt}")

    global pipe_mv, adapters
    load_pipe_mvadapter()
    imgs = MVAdapterT2MV.run_pipeline(
        pipe=pipe_mv,
        num_views=6,
        text=refined_prompt,
        height=768, width=768,
        num_inference_steps=50,
        guidance_scale=7.0,
        seed=seed,
        negative_prompt=negative_prompt,
        device="cuda",
        adapter_name_list=adapters,
    )
    # save and send to folder
    views_dir = _save_images_to_tmp(imgs)
    logger.info(f"[MV-ADAPTER]: generated {len(imgs)} views in {views_dir}")

    paths = sorted(glob.glob(os.path.join(views_dir, "*.png")))
    pipe_mv.to("cpu")
    del pipe_mv, adapters
    
    # llm quality checker and negative prompt generator
    llm_avaliation, llm_eval_neg_prompt = check_views_quality(views_dir, refined_prompt) # (refined_prompt, views_dir)
    return (
        paths,
        views_dir,
        gr.update(visible=True),
        refined_prompt,
        gr.update(value=llm_avaliation, visible=True),
        gr.update(value=llm_eval_neg_prompt, visible=True)
    )

def check_views_quality(views_dir, prompt):
    result = check_views(views_dir, prompt)
    llm_avaliation = result['avaliation']
    llm_neg_prompt = result['negative_prompt']

    logger.info(f"[MV-ADAPTER] views quality check: {result}")
    return llm_avaliation, llm_neg_prompt
