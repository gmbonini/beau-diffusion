import base64, io, os
import shutil
import tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from loguru import logger
import torch
from trellis_mv2m import TrellisMV2M

from mvadapter_t2mv import MVAdapterT2MV

logger.add("api.log", rotation="1 MB")
app = FastAPI(title="beau demo API")

def _pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    

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

    logger.info("[MV-ADAPTER] Pipeline loaded (API)")
    

@app.post("/t2mv/generate")
def t2mv_generate(ref_prompt, neg_prompt, randomize):
    seed = -1 if randomize else 42

    global pipe_mv, adapters
    load_pipe_mvadapter()
    
    imgs = MVAdapterT2MV.run_pipeline(
        pipe=pipe_mv,
        num_views=6,
        text=ref_prompt,
        height=768, width=768,
        num_inference_steps=50,
        guidance_scale=7.0,
        seed=seed,
        negative_prompt=neg_prompt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        adapter_name_list=adapters,
    )

    logger.info("[MV-ADAPTER] Images generated {} views (API)".format(len(imgs)))
    pipe_mv.to("cpu")
    del pipe_mv, adapters
    pipe_mv, adapters = None, []

    return {"views": [{"b64png": _pil_to_b64(im)} for im in imgs]}


# --------- TRELLIS

pipe_trellis = None
def load_pipe_trellis():
    global pipe_trellis
    pipe_trellis = TrellisMV2M(
        model="microsoft/TRELLIS-image-large",
        device="cuda"
    )
    logger.info("[TRELLIS] Pipeline loaded (API)")

    pipe_trellis.prepare_pipeline()
    logger.info("[TRELLIS] Pipeline prepared (API)")
    

@app.post("/mv2m/generate")
async def trellis_run_b64(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Files not found.")

    # uploadfiles > pil files
    views = []
    for uf in files:
        content = await uf.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        views.append(img)

    load_pipe_trellis()

    outputs = pipe_trellis.run_pipeline(views, seed=42)

    # tmp files
    outdir = tempfile.mkdtemp(prefix="trellis_")
    glb_path = os.path.join(outdir, "mesh.glb")
    ply_path = os.path.join(outdir, "mesh.ply")
    mp4_path = os.path.join(outdir, "preview.mp4")
    
    pipe_trellis.save_mesh(outputs, glb_path, ply_path, simplify=0.95, texture_size=1024)
    pipe_trellis.save_video(outputs, filename=mp4_path)
    
    logger.info("[TRELLIS] Mesh and video saved to tmp (API)")
    try:
        glb_b64 = _file_to_b64(glb_path)
        ply_b64 = _file_to_b64(ply_path)
        mp4_b64 = _file_to_b64(mp4_path)
        sizes = {
            "glb": os.path.getsize(glb_path),
            "ply": os.path.getsize(ply_path),
            "mp4": os.path.getsize(mp4_path),
        }
        logger.info("[TRELLIS] Sizes {} (API)".format(sizes))
        return {
            "glb_b64": glb_b64,
            "ply_b64": ply_b64,
            "mp4_b64": mp4_b64,
            "filenames": {"glb": "mesh.glb", "ply": "mesh.ply", "mp4": "preview.mp4"},
            "sizes": sizes,
        }
    finally:
        # clean tmp
        shutil.rmtree(outdir, ignore_errors=True)