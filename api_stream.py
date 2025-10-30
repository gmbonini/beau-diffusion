import base64, io, os
import shutil
import tempfile
import zipfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, BackgroundTasks
from fastapi.responses import FileResponse
from PIL import Image
from loguru import logger
import torch
from trellis_mv2m import TrellisMV2M
import cv2
from mvadapter_t2mv_sdxl import MVAdapterT2MV
import traceback
from ollama import start_ollama

# SQL
from sql.db_connector import DatabaseConnector, get_db_connector
import uuid

os.environ["ATTN_BACKEND"] = "xformers"

logger.add("api.log", rotation="1 MB")
app = FastAPI(title="beau demo API")

@app.on_event("startup")
def _startup():
    load_pipe_mvadapter()
    load_pipe_trellis()
    start_ollama()
    logger.info("[STARTUP] loaded pipelines")

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
    print('Loading MV-Adapter pipeline...')
    pipe_mv, adapters = MVAdapterT2MV.prepare_pipeline(
        base_model="Lykon/dreamshaper-xl-1-0", # Lykon/dreamshaper-xl-1-0 # stabilityai/stable-diffusion-xl-base-1.0
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        unet_model=None,
        lora_model= "goofyai/3d_render_style_xl/3d_render_style_xl.safetensors", # None,
        adapter_path="huanngzh/mv-adapter",
        scheduler= "lcm", # "ddpm",
        num_views=6,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    print('MV-Adapter pipeline loaded.')
    logger.info("[MV-ADAPTER] Pipeline loaded (API)")
    

@app.post("/t2mv/generate")
def t2mv_generate(ref_prompt, neg_prompt, randomize:bool = False):
    seed = -1 if randomize else 42
    seed = 1
    logger.info(f"[MV-ADAPTER] Generating images with seed {seed} (API)")
    logger.info(f"[MV-ADAPTER] prompt: {ref_prompt} and neg_prompt: {neg_prompt} (API)")
    
    global pipe_mv, adapters
    # load_pipe_mvadapter()
    
    imgs = MVAdapterT2MV.run_pipeline(
        pipe=pipe_mv,
        num_views=6,
        text=ref_prompt,
        height=768, width=768,
        num_inference_steps=12, # 50 default
        guidance_scale=7.0,
        seed=seed,
        negative_prompt=neg_prompt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        adapter_name_list=adapters,
        # azimuth_deg= [270]
        # azimuth_deg= [315]
        # azimuth_deg = [45, 180, 315]
       #  azimuth_deg = [0, 90, 270]
        # azimuth_deg = [0, 45, 90, 180, 270, 315]
    )

    logger.info("[MV-ADAPTER] Images generated {} views (API)".format(len(imgs)))
    # pipe_mv.to("cpu")-ls
    # del pipe_mv, adapters
    # pipe_mv, adapters = None, []

    return {"views": [{"b64png": _pil_to_b64(im)} for im in imgs]}

@app.post("/t2mv/generate2")
def t2mv_generate2(
    ref_prompt: str,
    neg_prompt: str,
    randomize: bool = False,
    background_tasks: BackgroundTasks = None
):
    seed = -1 if randomize else 42
    seed = 1
    logger.info(f"[MV-ADAPTER] Generating images with seed {seed} (API)")
    logger.info(f"[MV-ADAPTER] prompt: {ref_prompt} and neg_prompt: {neg_prompt} (API)")
    global pipe_mv, adapters
    # load_pipe_mvadapter()
    
    imgs = MVAdapterT2MV.run_pipeline(
        pipe=pipe_mv,
        num_views=6,
        text=ref_prompt,
        height=768, width=768,
        num_inference_steps=12, # 50 default
        guidance_scale=7.0,
        seed=seed,
        negative_prompt=neg_prompt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        adapter_name_list=adapters,
        # azimuth_deg= 45
        # azimuth_deg = [45, 180, 315]
        # azimuth_deg = [0, 90, 270]
        # azimuth_deg = [0, 45, 90, 180, 270, 315]
    )
    
    session_id = str(uuid.uuid4())
    zip_path = f"/tmp/views_{session_id}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, img in enumerate(imgs):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            zipf.writestr(f"view_{i:02d}.png", buf.getvalue())
    
    logger.info(f"[MV-ADAPTER] ZIP created with {len(imgs)} views")
    background_tasks.add_task(os.remove, zip_path)
    
    return FileResponse(
        zip_path, 
        media_type="application/zip",
        filename=f"views_{session_id}.zip",        
    )


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
    

import uuid

@app.post("/mv2m/generate")
async def trellis_run(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Files not found.")

    outdir = None
    try:
        # uploadfiles > pil files
        views = []
        for uf in files:
            content = await uf.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            views.append(img)

        # load_pipe_trellis()
        outputs = pipe_trellis.run_pipeline(views, seed=42)

        # tmp files with uuid
        session_id = str(uuid.uuid4())
        outdir = tempfile.mkdtemp(prefix=f"trellis_{session_id}_")
        glb_path = os.path.join(outdir, f"mesh_{session_id}.glb")
        ply_path = os.path.join(outdir, f"mesh_{session_id}.ply")
        mp4_path = os.path.join(outdir, f"preview_{session_id}.mp4")
                
        frame_save_dir = "temp_images_path"
        os.makedirs(frame_save_dir, exist_ok=True)
        frame_path = os.path.join(frame_save_dir, f"frame_{session_id}.jpg")
        
        pipe_trellis.save_mesh(outputs, glb_path, ply_path, simplify=0.95, texture_size=1024)
        pipe_trellis.save_video(outputs, filename=mp4_path)
        
        # extract middle frame
        cap = cv2.VideoCapture(mp4_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(frame_path, frame)
                logger.info(f"[TRELLIS] Frame saved on {frame_path}")
            else:
                logger.warning("[TRELLIS] Error saving frame")
            cap.release()
        else:
            logger.warning("[TRELLIS] Video not found for frame extraction")
        
        logger.info("[TRELLIS] Mesh and video saved to tmp (API)")        
        
        glb_b64 = _file_to_b64(glb_path)
        ply_b64 = _file_to_b64(ply_path)
        mp4_b64 = _file_to_b64(mp4_path)
        
        sizes = {
            "glb": os.path.getsize(glb_path),
            "ply": os.path.getsize(ply_path),
            "mp4": os.path.getsize(mp4_path),
        }
        if os.path.exists(frame_path):
            sizes["frame"] = os.path.getsize(frame_path)
            
        logger.info("[TRELLIS] Sizes {} (API)".format(sizes))
        
        result = {
            "glb_b64": glb_b64,
            "ply_b64": ply_b64,
            "mp4_b64": mp4_b64,
            "filenames": {
                "glb": f"mesh_{session_id}.glb",
                "ply": f"mesh_{session_id}.ply",
                "mp4": f"preview_{session_id}.mp4"
            },
            "sizes": sizes,
        }
        
        if os.path.exists(frame_path):
            result["frame_path"] = frame_path
            result["filenames"]["frame"] = f"frame_{session_id}.jpg"
            
        return result

    except Exception as e:
        logger.error("[TRELLIS] FAILED: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:

        if outdir and os.path.exists(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
      
@app.post("/feedback/save")
async def feedback_save(
    is_positive: int = Form(...),
    original_prompt: str = Form(...),
    refined: str = Form(None),
    chat: str = Form(None),
    negative_prompt: str = Form(None),
    video_frame_path: str = Form(None),
    multiview_files: List[UploadFile] = File(None),
    step: str = Form(None),
    db: DatabaseConnector = Depends(get_db_connector)
):
    """
    Save feedback and associated multiview images.
    Automatically cleans up oldest feedback if more than 100 multiview images exist.
    """
    if is_positive not in [0, 1]:
        raise HTTPException(status_code=422, detail="'is_positive' must be 0 or 1")

    try:
        
        multiview_count = db.get_multiview_count()
        
        
        new_images_count = len(multiview_files) if multiview_files else 0
        
        while multiview_count + new_images_count > 100:
            oldest_feedback = db.get_oldest_feedback_with_multiviews()
            if oldest_feedback:
                
                multiview_paths = db.get_multiview_paths(oldest_feedback['id'])
                for path in multiview_paths:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            logger.info(f"[CLEANUP] Deleted file: {path}")
                        except Exception as e:
                            logger.warning(f"[CLEANUP] Failed to delete {path}: {e}")
                
                
                db.delete_multiviews_by_feedback_id(oldest_feedback['id'])
                db.delete_feedback(oldest_feedback['id'])
                
                logger.info(f"[CLEANUP] Deleted oldest feedback (ID: {oldest_feedback['id']}) with {oldest_feedback['multiview_count']} images")
                
                
                multiview_count = db.get_multiview_count()
            else:
                break

        
        feedback_id = db.save_feedback(
            is_positive=is_positive,
            original_prompt=original_prompt,
            refined_prompt=refined,
            chat=chat,
            negative_prompt=negative_prompt,
            video_frame_url=video_frame_path,
            step=step
        )

        if multiview_files:
            temp_path = "temp_images_path"
            os.makedirs(temp_path, exist_ok=True)
                        
            batch_uuid = str(uuid.uuid4())
            
            for idx, uf in enumerate(multiview_files):
                file_ext = os.path.splitext(uf.filename)[1]
                unique_filename = f"{batch_uuid}_{idx}{file_ext}"
                file_path = os.path.join(temp_path, unique_filename)
                
                with open(file_path, "wb") as f:
                    f.write(await uf.read())
                db.save_multiview(feedback_id, file_path)

        logger.info(f"[FEEDBACK] Saved feedback (ID: {feedback_id}) with {new_images_count} images")
        return {"status": "success", "feedback_id": feedback_id}

    except Exception as e:
        logger.error(f"[DB ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))