import os, io, base64, requests
from PIL import Image
from mvadapter_t2mv import MVAdapterT2MV
from trellis_mv2m import TrellisMV2M
from ollama import refine_prompt, negative_prompt

import os, tempfile, glob, gc
import torch
from loguru import logger
import gradio as gr
from PIL import Image

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def _api_t2mv_generate(refined_prompt, negative_prompt, seed):
    resp = requests.post(
        f"{API_URL}/t2mv/generate",
        params={"prompt": refined_prompt, "negative_prompt": negative_prompt, "seed": seed},
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    views = data.get("views", [])
    if not views:
        raise RuntimeError(f"API retornou vazio: {data}")

    imgs = []
    for v in views:
        b64 = v.get("b64png")
        if not b64:
            continue
        raw = base64.b64decode(b64)
        imgs.append(Image.open(io.BytesIO(raw)).convert("RGB"))

    if not imgs:
        raise RuntimeError("[MV-Adapter] Invalid images.")
    return imgs


def generate_images(refined_prompt, negative_prompt, randomize=False):
    imgs = _api_t2mv_generate(refined_prompt, negative_prompt, randomize)

    views_dir = _save_images_to_tmp(imgs)
    logger.info(f"[MV-ADAPTER/API]: generated {len(imgs)} views in {views_dir}")

    paths = sorted(glob.glob(os.path.join(views_dir, "*.png")))

    llm_avaliation, llm_eval_neg_prompt = check_views_quality(views_dir, refined_prompt) # (refined_prompt, views_dir)

    return (
        paths,
        views_dir,
        gr.update(visible=True),
        refined_prompt,
        gr.update(value=llm_avaliation, visible=True),
        gr.update(value=llm_eval_neg_prompt, visible=True),
    )


# ------------- TRELLIS

def _b64_to_file(b64_str: str, path: str):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))

def generate_3d(views_dir):
    images_list, paths = _load_views_from_dir(views_dir)
    if len(paths) < 2:
        raise RuntimeError("Preciso de pelo menos 2 views para o TRELLIS.")

    # multipart with png
    files = [("files", (os.path.basename(p), open(p, "rb"), "image/png")) for p in paths]

    r = requests.post(f"{API_URL}/trellis/run_b64", files=files, timeout=900)
    r.raise_for_status()
    data = r.json()

    # save files on same dir
    glb_path = os.path.join(views_dir, data["filenames"]["glb"])
    ply_path = os.path.join(views_dir, data["filenames"]["ply"])
    mp4_path = os.path.join(views_dir, data["filenames"]["mp4"])

    _b64_to_file(data["glb_b64"], glb_path)
    _b64_to_file(data["ply_b64"], ply_path)
    _b64_to_file(data["mp4_b64"], mp4_path)

    logger.info(f"[TRELLIS/API-b64] assets: {glb_path}, {ply_path}, {mp4_path}")

    return (
        mp4_path,
        gr.update(value=glb_path, visible=True),
        gr.update(value=ply_path, visible=True),
    )