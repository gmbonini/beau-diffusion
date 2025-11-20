import base64, io, os
import shutil
import tempfile
import zipfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from loguru import logger
import torch
from trellis_mv2m import TrellisMV2M
from diffusers import FluxPipeline
import cv2
from mvadapter_t2mv_sdxl import MVAdapterT2MV
import traceback
from ollama import start_ollama, choose_generation_model
from concurrent.futures import ThreadPoolExecutor
import time
from transformers import AutoModelForImageSegmentation, AutoProcessor
from PIL import ImageDraw
from rembg import remove
from sql.db_connector import DatabaseConnector, get_db_connector
import uuid
import subprocess

os.environ["ATTN_BACKEND"] = "xformers"

logger.add("api.log", rotation="1 MB")
app = FastAPI(title="beau demo API")

@app.on_event("startup")
def _startup():
    load_pipe_flux_t2i()
    load_pipe_mvadapter()
    load_pipe_trellis()
    start_ollama()
    logger.info("[STARTUP] loaded pipelines")

def _file_to_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to convert file {path} to b64: {e}")
        return ""

# MV-Adapter

pipe_mv, adapters = None, []
def load_pipe_mvadapter():
    global pipe_mv, adapters
    print('Loading MV-Adapter pipeline...')
    try:
        pipe_mv, adapters = MVAdapterT2MV.prepare_pipeline(
            base_model="SG161222/RealVisXL_V5.0",
            vae_model="madebyollin/sdxl-vae-fp16-fix",
            unet_model=None,
            lora_model=None,
            adapter_path="huanngzh/mv-adapter",
            scheduler="dpmpp_2m",
            num_views=6,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
        )
        try:
            print('Loading 3D Render LoRA (artificialguybr/3DRedmond-V1)...')
            pipe_mv.load_lora_weights(
                "artificialguybr/3DRedmond-V1",
                weight_name="3DRedmond-3DRenderStyle-3DRenderAF.safetensors"
            )
            pipe_mv.fuse_lora(lora_scale=0.75)
            print('3D Render LoRA (artificialguybr) loaded and fused.')
            logger.info("[MV-ADAPTER] 3D Render LoRA (artificialguybr) loaded with scale 0.75")
        except Exception as e:
            logger.error(f"[MV-ADAPTER] Failed to load LoRA (artificialguybr): {e}")
            print(f"Warning: Could not load artificialguybr LoRA - {e}")


        print('MV-Adapter pipeline loaded.')
        logger.info("[MV-ADAPTER] Pipeline loaded (API)")
        
    except Exception as e:
        logger.error(f"[MV-ADAPTER] CRITICAL FAIL loading pipeline: {e}\n{traceback.format_exc()}")
        pipe_mv = None

# RMBG

pipe_rmbg_model = None
pipe_rmbg_processor = None
rmbg_device = "cuda" if torch.cuda.is_available() else "cpu"
        
def _remove_bg(image: Image.Image) -> Image.Image:
            
    if image.mode == 'RGBA':
        image = image.convert("RGB")

    try:
        
        final_image = remove(
            image,
            alpha_matting=True,
            alpha_matting_erode_size=5
        )
        return final_image

    except Exception as e:
        logger.warning(f"[RMBG] Failed to process image with 'rembg' lib: {e}. Returning original.")        
        return image

# MV-Adapter

@app.post("/t2mv/generate")
def t2mv_generate(
    ref_prompt: str, 
    neg_prompt: str = "", 
    randomize: bool = False, 
    inference_steps: int = 28,
    use_3d_trigger: bool = True,
    background_tasks: BackgroundTasks = None
):
    try:
        start_time = time.time()
        seed = -1 if randomize else 42
        
        enhanced_prompt = ref_prompt
        if use_3d_trigger:
            enhanced_prompt = f"3d style, 3d, {ref_prompt}, game asset"
        
        enhanced_negative = f"{neg_prompt}, blurry, distorted, deformed, ugly, bad geometry, bad topology, watermark, signature, text, words, letters, nsfw, explicit content" if neg_prompt else "realistic photo, photorealistic, blurry, distorted, deformed, ugly, bad geometry, watermark, text, words, letters, nsfw, explicit content"
        
        logger.info(f"[MV-ADAPTER] Generating images with seed {seed} and {inference_steps} steps (API)")
        logger.info(f"[MV-ADAPTER] Enhanced prompt: {enhanced_prompt}")
        
        global pipe_mv, adapters
        
        imgs = MVAdapterT2MV.run_pipeline(
            pipe=pipe_mv,
            num_views=6,
            text=enhanced_prompt,
            height=768, width=768,
            num_inference_steps=inference_steps,
            guidance_scale=5.5,
            seed=seed,
            negative_prompt=enhanced_negative,
            device="cuda" if torch.cuda.is_available() else "cpu",
            adapter_name_list=adapters,
        )
        
        logger.info(f"[MV-ADAPTER] Images generated with {inference_steps} steps - {len(imgs)} views (API)")
        
        logger.info(f"[MV-ADAPTER] Images generated, encoding to base64...")
        
        b64_images = []
        for img in imgs:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            b64_images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        elapsed_time = time.time() - start_time
        logger.info(f"[MV-ADAPTER] Total time (with encoding): {elapsed_time:.2f}s")
        
        
        return {"images": b64_images}
    except Exception as e:
        logger.error(f"[MV-ADAPTER] FAILED: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating MV images: {str(e)}")

# Trellis

pipe_trellis = None
def load_pipe_trellis():
    global pipe_trellis
    try:
        pipe_trellis = TrellisMV2M(
            model="microsoft/TRELLIS-image-large",
            device="cuda"
        )
        logger.info("[TRELLIS] Pipeline loaded (API)")

        pipe_trellis.prepare_pipeline()
        logger.info("[TRELLIS] Pipeline prepared (API)")
    except Exception as e:
        logger.error(f"[TRELLIS] Failed to load pipeline: {e}\n{traceback.format_exc()}")
        pipe_trellis = None
    

@app.post("/mv2m/generate")
async def trellis_run(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Files not found.")

    outdir = None
    t_total_start = time.time()
    logger.info("[TRELLIS] --- START REQUEST ---")

    try:
        t_step = time.time()
        views = []

        logger.info("[TRELLIS] Starting input decode + BG removal")
        for uf in files:
            t_file = time.time()
            content = await uf.read()
            img = Image.open(io.BytesIO(content))
            logger.info(f"[TRELLIS] Loaded image {uf.filename} in {time.time()-t_file:.3f}s")

            t_bg = time.time()
            img_no_bg = _remove_bg(img)            
            logger.info(f"[TRELLIS] Removed BG in {time.time()-t_bg:.3f}s")

            views.append(img_no_bg)

        logger.info(f"[TRELLIS] Input preprocessing finished in {time.time()-t_step:.3f}s")
        logger.info(f"[TRELLIS] Total views: {len(views)}")

        t_step = time.time()
        logger.info("[TRELLIS] Running Trellis pipeline...")
        outputs = pipe_trellis.run_pipeline(views, seed=42)
        logger.info(f"[TRELLIS] Trellis pipeline duration: {time.time()-t_step:.3f}s")

        t_step = time.time()
        session_id = str(uuid.uuid4())
        outdir = tempfile.mkdtemp(prefix=f"trellis_{session_id}_")
        logger.info(f"[TRELLIS] Output directory prepared in {time.time()-t_step:.3f}s")


        glb_filename = f"mesh_{session_id}.glb"
        ply_filename = f"mesh_{session_id}.ply"
        mp4_filename = f"preview_{session_id}.mp4"
        frame_filename = f"frame_{session_id}.jpg"

        glb_path = os.path.join(outdir, glb_filename)
        ply_path = os.path.join(outdir, ply_filename)
        mp4_path = os.path.join(outdir, mp4_filename)
        frame_path = os.path.join(outdir, frame_filename)

        t_step = time.time()
        logger.info("[TRELLIS] Starting save_video...")
        _, _, middle_frame_np = pipe_trellis.save_video(outputs, filename=mp4_path)
        logger.info(f"[TRELLIS] save_video completed in {time.time()-t_step:.3f}s")

        t_step = time.time()
        logger.info("[TRELLIS] Starting save_mesh (this can be slow)...")
        pipe_trellis.save_mesh(
            outputs, 
            glb_path=glb_path, 
            ply_path=ply_path,
            simplify=0.80,
            texture_size=1024
        )
        logger.info(f"[TRELLIS] save_mesh completed in {time.time()-t_step:.3f}s")

        t_step = time.time()
        try:
            logger.info("[TRELLIS] Saving middle frame...")
            middle_frame_bgr = cv2.cvtColor(middle_frame_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, middle_frame_bgr)
            logger.info(f"[TRELLIS] Frame saved in {time.time()-t_step:.3f}s")
        except Exception as e:
            logger.warning(f"[TRELLIS] Frame save ERROR in {time.time()-t_step:.3f}s: {e}")

        t_step = time.time()
        logger.info("[TRELLIS] Creating ZIP buffer...")
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_STORED) as zipf:
            for file_path, file_name in [
                (glb_path, glb_filename),
                (ply_path, ply_filename),
                (mp4_path, mp4_filename),
                (frame_path, frame_filename)
            ]:
                if os.path.exists(file_path):
                    t_zip_item = time.time()
                    zipf.write(file_path, arcname=file_name)
                    logger.info(f"[TRELLIS] Added {file_name} (took {time.time()-t_zip_item:.3f}s)")
                else:
                    logger.error(f"[TRELLIS] MISSING: {file_name} not found!")

        logger.info(f"[TRELLIS] ZIP creation time: {time.time()-t_step:.3f}s")

        t_step = time.time()
        zip_buffer.seek(0)
        zip_buffer_check = io.BytesIO(zip_buffer.getvalue())

        with zipfile.ZipFile(zip_buffer_check, 'r') as zipf:
            logger.info(f"[TRELLIS] ZIP contains: {zipf.namelist()}")

        logger.info(f"[TRELLIS] ZIP validation took {time.time()-t_step:.3f}s")

        total_time = time.time() - t_total_start
        logger.info(f"[TRELLIS] --- FINISHED REQUEST in {total_time:.3f}s ---")

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=trellis_output_{session_id}.zip"}
        )

    except Exception as e:
        logger.error("[TRELLIS] FAILED: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        t_cleanup = time.time()
        if outdir and os.path.exists(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
        logger.info(f"[TRELLIS] Cleanup finished in {time.time()-t_cleanup:.3f}s")


# Save feedback

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
    negative_feedback: str = Form(None),
    db: DatabaseConnector = Depends(get_db_connector)
):
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
            step=step,
            negative_feedback=negative_feedback
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

# Flux

pipe_flux_t2i = None

def load_pipe_flux_t2i():
    global pipe_flux_t2i
    
    if not torch.cuda.is_available():
        logger.warning("[FLUX-T2I] CUDA not available, skipping FLUX load.")
        return

    logger.info("[FLUX-T2I] Loading FLUX.1-schnell pipeline...")
    
    try:
        
        pipe_flux_t2i = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16
        ).to("cuda")
        
        logger.info("[FLUX-T2I] FLUX.1-schnell pipeline loaded.")
        
        # try:
        #     logger.info("[FLUX-T2I] Loading Hyper-SD LoRA for speedup (ByteDance/Hyper-SD)...")
        #     pipe_flux_t2i.load_lora_weights(
        #         "ByteDance/Hyper-SD",
        #         weight_name="Hyper-FLUX.1-schnell-8steps-lora.safetensors"
        #     )
        #     pipe_flux_t2i.fuse_lora(lora_scale=0.125)
        #     logger.info("[FLUX-T2I] Hyper-SD LoRA loaded and fused (scale=0.125)")
        # except Exception as e:
        #     logger.error(f"[FLUX-T2I] Failed to load Hyper-SD LoRA: {e}")

        
        # try:
        #     logger.info("[FLUX-T2I] Loading LoRA: gokaygokay/Flux-Game-Assets-LoRA-v2")
        #     pipe_flux_t2i.load_lora_weights(
        #         "gokaygokay/Flux-Game-Assets-LoRA-v2",
        #         weight_name="game_asst.safetensors"
        #     )
            
            
        #     pipe_flux_t2i.fuse_lora(lora_scale=0.8) 
        #     logger.info("[FLUX-T2I] LoRA 'Flux-Game-Assets-LoRA-v2' loaded and fused with scale 0.8.")
        # except Exception as e:
            
        #     logger.error(f"[FLUX-T2I] Failed to load/fuse Game Assets LoRA: {e}\n{traceback.format_exc()}")
        

    except Exception as e:
        logger.error(f"[FLUX-T2I] Failed to load FLUX pipeline: {e}\n{traceback.format_exc()}")
        pipe_flux_t2i = None 

@app.post("/t2i/generate_flux")
def t2i_generate_flux(
    ref_prompt: str, 
    neg_prompt: str = "",
    randomize: bool = False, 
    inference_steps: int = 10
):
    
    global pipe_flux_t2i
    if pipe_flux_t2i is None:
        logger.error("[FLUX-T2I] Pipeline não está carregado!")
        raise HTTPException(status_code=503, detail="FLUX pipeline is not loaded or failed to load.")

    start_time = time.time()
    seed = -1 if randomize else 42
    
    generator = torch.Generator(device="cuda").manual_seed(seed) if not randomize else None
        
    enhanced_prompt = f"wbgmsst, {ref_prompt}, white background, no text, no words, no letters, isometric view, 3d render, game asset"
    
    logger.info(f"[FLUX-T2I] Generating image with seed {seed} and {inference_steps} steps")
   
    logger.info(f"[FLUX-T2I] Enhanced Prompt: {enhanced_prompt}")

    try:
        result = pipe_flux_t2i(
            prompt=enhanced_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=3.5,
            generator=generator           
        )
        img = result.images[0]
        
    except Exception as e:
        logger.error("[FLUX-T2I] FAILED: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during FLUX generation: {e}")

    logger.info(f"[FLUX-T2I] Image generated, encoding to base64...")
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    elapsed_time = time.time() - start_time
    logger.info(f"[FLUX-T2I] Total time (with encoding): {elapsed_time:.2f}s")
    
    return {"image": b64_image}

# Ollama choose model

@app.post("/t2g/choose_model")
def t2g_choose_model(prompt: str = Form(...)):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    
    try:
        choice = choose_generation_model(prompt)
        return {"model_choice": choice}
    except Exception as e:
        logger.error(f"[API /t2g/choose_model] FAILED: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))