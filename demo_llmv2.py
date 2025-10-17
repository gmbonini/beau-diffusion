from mvadapter_t2mv import MVAdapterT2MV
from trellis_mv2m import TrellisMV2M
from ollama import refine_prompt, negative_prompt, check_views, prepare_prompts

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
def llm_prompt_processing(prompt):
    # ref_prompt = refine_prompt(prompt)
    # neg_prompt = negative_prompt(ref_prompt)
    logger.info(f"[MV-ADAPTER] generating images. prompt to be refined: {prompt}")
    result = prepare_prompts(prompt)
    ref_prompt = result['refined']
    neg_prompt = result['negative'] or "nothing to change"
    logger.info(f"[MV-ADAPTER] refined prompt: {ref_prompt}")
    logger.info(f"[MV-ADAPTER] negative prompt: {neg_prompt}")
    return (
        gr.update(value=ref_prompt, visible=True), # ref_prompt
        gr.update(value=neg_prompt, visible=True), # neg_prompt
        ref_prompt, # state_last_prompt
        neg_prompt # state_neg_prompt
    )

def generate_images(refined_prompt, negative_prompt, randomize=False):
    seed = -1 if randomize else 42

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
    llm_avaliation, llm_eval_neg_prompt = check_views_quality(refined_prompt, views_dir)
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

    glb_path = os.path.join(views_dir, "mesh.glb")
    ply_path = os.path.join(views_dir, "mesh.ply")
    out_mp4 = os.path.join(views_dir, "multiview_gradio.mp4")

    glb_filename, ply_filename = pipe_trellis.save_mesh(outputs, glb_path, ply_path, simplify=0.95, texture_size=1024)
    logger.info(f"[TRELLIS] saved glb to {glb_path} and ply to {ply_path}")
    video_out, video_filename = pipe_trellis.save_video(outputs, filename=out_mp4)

    del pipe_trellis
    return video_filename, gr.update(value=glb_filename, visible=True), gr.update(value=ply_filename, visible=True),

 # ------------ UI helpers ------------
def _lock_all_buttons():
    #
    return (
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )

def _unlock_after_generate():
    # run + remake + trellis
    return (
        gr.update(interactive=True),# run_btn
        gr.update(visible=True, interactive=True), # remake_btn
        gr.update(visible=True, interactive=True), # run_trellis_btn
    )

def _unlock_after_trellis():
    # all hab
    return (
        gr.update(interactive=True), # run_btn
        gr.update(visible=True, interactive=True),# remake_btn
        gr.update(visible=True, interactive=True), # run_trellis_btn
    )


with gr.Blocks() as demo:
    gr.Markdown("llm demo")

    # images to generate
    # state_images = gr.State([])
    state_views_dir = gr.State(value="")
    state_last_prompt = gr.State(value="")
    state_neg_prompt = gr.State(value="")

    prompt = gr.Textbox(label="Prompt")
    run_btn = gr.Button("Generate")

    with gr.Accordion("LLM prompts", open=True):
        ref_prompt = gr.Textbox(
            label="Refined positive prompt",
            interactive=False,
            visible=False,
            show_copy_button=True,
            lines=2
        )
        neg_prompt = gr.Textbox(
            label="Negative prompt",
            interactive=False,
            visible=False,
            show_copy_button=True,
            lines=2
        )

    remake_btn = gr.Button("Remake images (same prompt but different seed)", visible=False)
    gallery = gr.Gallery(columns=3, rows=2, height="auto")

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

    # only shows when mv adapter ends
    run_trellis_btn = gr.Button("Generate 3D Mesh", visible=False)
    video = gr.Video(label="Mesh video preview")


    # download_glb = gr.File(label="Download GLB mesh", visible=False)
    # download_ply = gr.File(label="Download PLY mesh", visible=False)
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


    # generate
    # GENERATE V3 - SHOW LLM FIRST
    #first step show llm
    run_btn.click(
    fn=_lock_all_buttons, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn]
    ).then(
    fn=llm_prompt_processing,
    inputs=prompt,
    outputs=[ref_prompt, neg_prompt, state_last_prompt, state_neg_prompt],
    ).then(
    fn=generate_images,
    inputs=[state_last_prompt, state_neg_prompt, gr.State(False)],
    outputs=[gallery, state_views_dir, run_trellis_btn, state_last_prompt, llm_avaliation, llm_eval_neg_prompt],
    queue=True
    ).then(
        fn=_unlock_after_generate, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn]
    )
    # GENERATE V2
    # run_btn.click(
    # fn=_lock_all_buttons, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn]
    # ).then(
    #     fn=llm_prompt_processing, inputs=prompt,
    #     outputs=[ref_prompt, neg_prompt, state_last_prompt, state_neg_prompt],
    # ).then(
    #     fn=generate_images, inputs=[state_last_prompt, state_neg_prompt, gr.State(False)],
    #     outputs=[gallery, state_views_dir, run_trellis_btn, state_last_prompt, ref_prompt, neg_prompt, llm_avaliation, llm_eval_neg_prompt]
    # ).then(
    #     fn=_unlock_after_generate, inputs=None,
    #     outputs=[run_btn, remake_btn, run_trellis_btn]
    # )
    # GENERATE V1
    # run_btn.click(
    #     fn=_lock_all_buttons, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn]
    # ).then(
    #     fn=generate_images, inputs=[prompt, gr.State(False)],
    #     outputs=[gallery, state_views_dir, run_trellis_btn, state_last_prompt, ref_prompt, neg_prompt, llm_avaliation, llm_eval_neg_prompt]
    # ).then(
    #     fn=_unlock_after_generate, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn]
    # )

    # remake
    remake_btn.click(
    fn=_lock_all_buttons, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn]
    ).then(
        fn=generate_images, inputs=[state_last_prompt, state_neg_prompt, gr.State(True)],
        outputs=[gallery, state_views_dir, run_trellis_btn, state_last_prompt, llm_avaliation, llm_eval_neg_prompt]
    ).then(
        fn=_unlock_after_generate, inputs=None,
        outputs=[run_btn, remake_btn, run_trellis_btn]
    )

    # remake_btn.click(
    #     fn=_lock_all_buttons, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn]
    # ).then(
    #     fn=generate_images, inputs=[state_last_prompt, gr.State(True)],
    #     outputs=[gallery, state_views_dir, run_trellis_btn, state_last_prompt, ref_prompt, neg_prompt, llm_avaliation, llm_eval_neg_prompt]
    # ).then(
    #     fn=_unlock_after_generate, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn]
    # )

    # trellis
    run_trellis_btn.click(
        fn=_lock_all_buttons, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn]
    ).then(
        fn=generate_3d, inputs=state_views_dir, outputs=[video, download_glb, download_ply]
    ).then(
        fn=_unlock_after_trellis, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn]
    )

    # # trellis
    # run_trellis_btn.click(
    #     fn=_lock_all_buttons, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn]
    # ).then(
    #     fn=generate_3d, inputs=state_views_dir, outputs=[video, download_glb, download_ply]
    # ).then(
    #     fn=_unlock_after_trellis, inputs=None, outputs=[run_btn, remake_btn, run_trellis_btn]
    # )


    # btns

    # # SECOND vers
    # run_btn.click(
    #     fn=generate_images,
    #     inputs=[prompt, gr.State(False)],
    #     outputs=[gallery, state_views_dir, run_trellis_btn, state_last_prompt],
    # ).then(  #show remake btn
    #     fn=lambda: gr.update(visible=True), inputs=None, outputs=remake_btn
    # )

    # remake_btn.click(
    #     fn=generate_images,
    #     inputs=[state_last_prompt, gr.State(True)],
    #     outputs=[gallery, state_views_dir, run_trellis_btn, state_last_prompt],
    # )

    # run_trellis_btn.click(
    #     fn=generate_3d,
    #     inputs=state_views_dir,
    #     outputs=video,
    # )


    # FIRST VER
    # run_btn.click(
    #     fn=generate_images,
    #     inputs=prompt,
    #     outputs=[gallery, state_views_dir, run_trellis_btn],
    # )

    # remake_btn.click(
    #     fn=generate_images,
    #     inputs=[state_last_prompt, gr.State(True)],
    #     outputs=[gallery, state_views_dir, run_trellis_btn, state_last_prompt],
    # )

    # run_trellis_btn.click(
    #     fn=generate_3d,
    #     inputs=state_views_dir,
    #     outputs=video,
    # )

if __name__ == "__main__":
    # load_pipe_mvadapter()
    # load_pipe_trellis()
    demo.launch(share=True)