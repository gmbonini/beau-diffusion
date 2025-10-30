import argparse

import torch
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
import sys
import os

MVADAPTER_PATH = "/workspace/beau/MV-Adapter"
if MVADAPTER_PATH not in sys.path:
    sys.path.append(MVADAPTER_PATH)

from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import make_image_grid
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
from mvadapter.utils.mesh_utils import get_orthogonal_camera

class MVAdapterT2MV:
    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vae_model: str | None = "madebyollin/sdxl-vae-fp16-fix",
        unet_model: str | None = None,
        lora_model: str | None = None,
        adapter_path: str = "huanngzh/mv-adapter",
        scheduler: str | None = None,
        num_views: int = 6,
        device: str = "cuda",
        dtype: torch.dtype | str = "auto",
        ):
        self.base_model = base_model
        self.vae_model = vae_model
        self.unet_model = unet_model
        self.lora_model = lora_model
        self.adapter_path = adapter_path
        self.scheduler = scheduler
        self.num_views = num_views
        self.device = device

    def prepare_pipeline(
        base_model,
        vae_model,
        unet_model,
        lora_model,
        adapter_path,
        scheduler,
        num_views,
        device,
        dtype,
    ):
        # Load vae and unet if provided
        pipe_kwargs = {}
        if vae_model is not None:
            pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype)
        if unet_model is not None:
            pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model,torch_dtype=dtype)

        # Prepare pipeline
        pipe: MVAdapterT2MVSDXLPipeline
        pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(
            base_model, 
            torch_dtype=dtype,
            **pipe_kwargs
        )

        # Load scheduler if provided
        scheduler_class = None
        if scheduler == "ddpm":
            scheduler_class = DDPMScheduler
        elif scheduler == "lcm":
            scheduler_class = LCMScheduler

        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=scheduler_class,
        )
        pipe.init_custom_adapter(num_views=num_views)
        pipe.load_custom_adapter(
            adapter_path, weight_name="mvadapter_t2mv_sdxl.safetensors"
        )

        pipe.to(device=device, dtype=dtype)
        pipe.cond_encoder.to(device=device, dtype=dtype)

        # load lora if provided
        adapter_name_list = []
        if lora_model is not None:
            lora_model_list = lora_model.split(",")
            for lora_model_ in lora_model_list:
                model_, name_ = lora_model_.strip().rsplit("/", 1)
                adapter_name = name_.split(".")[0]
                adapter_name_list.append(adapter_name)
                pipe.load_lora_weights(model_, weight_name=name_, adapter_name=adapter_name)

        # vae slicing for lower memory usage
        pipe.enable_vae_slicing()

        return pipe, adapter_name_list


    def run_pipeline(
        pipe,
        num_views,
        text,
        height,
        width,
        num_inference_steps,
        guidance_scale,
        seed,
        negative_prompt,
        lora_scale=["1.0"],
        device="cuda",
        azimuth_deg=None,
        adapter_name_list=[],
    ):
        # Set lora scale
        if len(adapter_name_list) > 0:
            if len(lora_scale) == 1:
                lora_scale = [lora_scale[0]] * len(adapter_name_list)
            else:
                assert len(lora_scale) == len(
                    adapter_name_list
                ), "Number of lora scales must match number of adapters"
            lora_scale = [float(s) for s in lora_scale]
            pipe.set_adapters(adapter_name_list, adapter_weights=lora_scale)
            print(f"Loaded {len(adapter_name_list)} adapters with scales {lora_scale}")

        # Prepare cameras
        if azimuth_deg is None:
            azimuth_deg = [0, 45, 90, 180, 270, 315]
        cameras = get_orthogonal_camera(
            elevation_deg=[0] * num_views,
            distance=[1.8] * num_views,
            left=-0.55,
            right=0.55,
            bottom=-0.55,
            top=0.55,
            azimuth_deg=[x - 90 for x in azimuth_deg],
            device=device,
        )

        plucker_embeds = get_plucker_embeds_from_cameras_ortho(
            cameras.c2w, [1.1] * num_views, width
        )
        control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

        pipe_kwargs = {"max_sequence_length": 214}
        if seed != -1:
            pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

        images = pipe(
            text,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            negative_prompt=negative_prompt,
            **pipe_kwargs,
        ).images

        return images
