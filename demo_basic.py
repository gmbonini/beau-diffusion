from mvadapter_t2mv import MVAdapterT2MV
from trellis_mv2m import TrellisMV2M
import os, tempfile, glob, gc
import torch
from loguru import logger
import gradio as gr
from PIL import Image


logger.add("gradio.log", rotation="1 MB")

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

# def generate_images(prompt):
#     global pipe_mv, adapters
#     load_pipe_mvadapter()
#     imgs = MVAdapterT2MV.run_pipeline(
#         pipe=pipe_mv,
#         num_views=6,
#         text=prompt,
#         height=768, width=768,
#         num_inference_steps=50,
#         guidance_scale=7.0,
#         seed=42,
#         negative_prompt="ugly anatomy",
#         device="cuda",
#         adapter_name_list=adapters,
#     )

#     pipe_mv.to("cpu")
#     del pipe_mv, adapters
#     return imgs, imgs, gr.update(visible=True)
def generate_images(prompt):
    global pipe_mv, adapters
    load_pipe_mvadapter()
    imgs = MVAdapterT2MV.run_pipeline(
        pipe=pipe_mv,
        num_views=6,
        text=prompt,
        height=768, width=768,
        num_inference_steps=50,
        guidance_scale=7.0,
        seed=42,
        negative_prompt="ugly anatomy",
        device="cuda",
        adapter_name_list=adapters,
    )
    # save and send to folder
    views_dir = _save_images_to_tmp(imgs)
    logger.info(f"[MV-ADAPTER]: generated {len(imgs)} views in {views_dir}")

    paths = sorted(glob.glob(os.path.join(views_dir, "*.png")))
    pipe_mv.to("cpu")
    del pipe_mv, adapters
    return paths, views_dir, gr.update(visible=True)


 # ------------ TRELLIS ------------
pipe_trellis = None
def load_pipe_trellis():
    global pipe_trellis
    pipe_trellis = TrellisMV2M(
        model="microsoft/TRELLIS-image-large",
        device="cuda"
    )

    pipe_trellis.prepare_pipeline()
    logger.info("[TRELLIS] Pipeline loaded")

def generate_3d(views_dir):
    # global pipe_trellis
    # load_pipe_trellis()
    # outputs = pipe_trellis.run_pipeline(images, seed=42)
    # video, filename = pipe_trellis.save_video(outputs, filename="multiview_gradio.mp4")
    # # mesh, filename_mesh = pipe_trellis.save_mesh(outputs, filename="multiview_gradio.obj") # need to check

    # # pipe_trellis.to("cpu")
    # del pipe_trellis
    # return video #, filename_mesh
    
    # paths = sorted(p for p in (os.path.join(images, f) for f in os.listdir(images)) if p.endswith(".png"))
    # images = [Image.open(p).convert("RGB") for p in paths]
    # outputs = pipe_trellis.run_pipeline(images, seed=42)
    # video_path = pipe_trellis.save_video(outputs, filename=os.path.join(images, "multiview_gradio.mp4"))

    global pipe_trellis
    load_pipe_trellis()
    images_list, paths = _load_views_from_dir(views_dir)
    logger.info(f"[TRELLIS] received {len(images_list)} views: {paths}")
    
    assert len(images_list) >= 2, "need at least 2 views to generate 3D"
    outputs = pipe_trellis.run_pipeline(images_list, seed=42)

    out_mp4 = os.path.join(views_dir, "multiview_gradio.mp4")
    video_out, filename = pipe_trellis.save_video(outputs, filename=out_mp4)

    del pipe_trellis
    return filename


with gr.Blocks() as demo:
    gr.Markdown("basic demo")

    # images to generate
    # state_images = gr.State([])
    state_views_dir = gr.State(value="")

    prompt = gr.Textbox(label="Prompt")
    run_btn = gr.Button("Generate")
    gallery = gr.Gallery(columns=3, rows=2, height="auto")

    # only shows when mv adapter ends
    run_trellis_btn = gr.Button("Generate 3D Mesh video", visible=False)
    video = gr.Video(label="3D Video")

    # btns
    run_btn.click(
        fn=generate_images,
        inputs=prompt,
        outputs=[gallery, state_views_dir, run_trellis_btn],
    )

    run_trellis_btn.click(
        fn=generate_3d,
        inputs=state_views_dir,
        outputs=video,
    )

if __name__ == "__main__":
    # load_pipe_mvadapter()
    # load_pipe_trellis()
    demo.launch(share=True)