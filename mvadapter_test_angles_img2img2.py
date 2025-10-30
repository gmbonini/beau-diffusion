# for some reason, this code is faster

import torch
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
import sys
import os
import time  
from PIL import Image  
 
MVADAPTER_PATH = "/workspace/beau/MV-Adapter"
if MVADAPTER_PATH not in sys.path:
    sys.path.append(MVADAPTER_PATH)
 
from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
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
        adapter_path: str = "huanngzh/mv-adapter",
        scheduler: str | None = None,
        num_views: int = 6,
        device: str = "cuda",
        dtype: torch.dtype | str = "auto",
    ):
        self.base_model = base_model
        self.vae_model = vae_model
        self.unet_model = unet_model
        self.adapter_path = adapter_path
        self.scheduler = scheduler
        self.num_views = num_views 
        self.device = device
        self.dtype = dtype

        self.vae, self.unet, scheduler_class_enum = self._prepare_components(
            vae_model, unet_model, scheduler, device, dtype
        )
        
        self.pipe = self.load_pipeline(
            MVAdapterT2MVSDXLPipeline, 
            base_model,
            adapter_path,
            num_views,
            device,
            dtype,
            self.vae,
            self.unet,
            scheduler_class_enum
        )
 
    def _prepare_components(self, vae_model, unet_model, scheduler, device, dtype):
        vae = None
        unet = None
        
        if vae_model is not None:
            vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype) 
        if unet_model is not None:
            unet = UNet2DConditionModel.from_pretrained(unet_model, torch_dtype=dtype) 
        
        scheduler_class_enum = None
        if scheduler == "ddpm":
            scheduler_class_enum = DDPMScheduler
        elif scheduler == "lcm":
            scheduler_class_enum = LCMScheduler
            
        return vae, unet, scheduler_class_enum

    def load_pipeline(
        self,
        pipeline_class, 
        base_model,
        adapter_path,
        num_views,
        device,
        dtype,
        vae, 
        unet, 
        scheduler_class_enum 
    ):
        
        pipe_kwargs = {}
        if vae:
            pipe_kwargs["vae"] = vae
        if unet:
            pipe_kwargs["unet"] = unet
        
        pipe = pipeline_class.from_pretrained(base_model, torch_dtype=dtype, **pipe_kwargs) 
 
        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=scheduler_class_enum,
        )
        pipe.init_custom_adapter(num_views=num_views)
        pipe.load_custom_adapter(
            adapter_path, weight_name="mvadapter_t2mv_sdxl.safetensors"
        )
 
        pipe.to(device=device, dtype=dtype) 
        pipe.cond_encoder.to(device=device, dtype=dtype)
 
        pipe.enable_vae_slicing()
 
        return pipe
 
    def run_pipeline(
        self,
        text,
        height,
        width,
        num_inference_steps,
        guidance_scale,
        seed,
        negative_prompt,
        azimuth_deg=None,
    ):        
        
        pipe = self.pipe
        device = self.device
        
        if azimuth_deg is None:
            azimuth_deg = [0, 45, 90, 180, 270, 315]
        
        num_views_to_generate = len(azimuth_deg)
 
        cameras = get_orthogonal_camera(
            elevation_deg=[0] * num_views_to_generate,
            distance=[1.8] * num_views_to_generate,
            left=-0.55, right=0.55, bottom=-0.55, top=0.55,
            azimuth_deg=azimuth_deg,
            device=device,
        )
 
        plucker_embeds = get_plucker_embeds_from_cameras_ortho(
            cameras.c2w, [1.1] * num_views_to_generate, width
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
            num_images_per_prompt=num_views_to_generate, 
            control_image=control_images,
            control_conditioning_scale=1.0,
            negative_prompt=negative_prompt,
            **pipe_kwargs,
        ).images
 
        return images


class MVAdapterI2MV:  
    
    def __init__(self, t2mv_model: MVAdapterT2MV):
        print("\nCarregando pipeline I2MV...")
        self.device = t2mv_model.device
        self.dtype = t2mv_model.dtype
                
        vae = t2mv_model.pipe.vae
        image_encoder = t2mv_model.pipe.image_encoder
        text_encoder = t2mv_model.pipe.text_encoder
        text_encoder_2 = t2mv_model.pipe.text_encoder_2
        tokenizer = t2mv_model.pipe.tokenizer
        tokenizer_2 = t2mv_model.pipe.tokenizer_2
                
        scheduler_class_enum = None
        if t2mv_model.scheduler == "ddpm":
            scheduler_class_enum = DDPMScheduler
        elif t2mv_model.scheduler == "lcm":
            scheduler_class_enum = LCMScheduler    
        
        base_scheduler_config = t2mv_model.pipe.scheduler.config        
        clean_base_scheduler = scheduler_class_enum.from_config(base_scheduler_config)
        
        clean_wrapped_scheduler = ShiftSNRScheduler.from_scheduler(
            clean_base_scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=scheduler_class_enum,
        )
                    
        unet_path = t2mv_model.unet_model if t2mv_model.unet_model else t2mv_model.base_model
        unet_kwargs = {"torch_dtype": self.dtype}
        if t2mv_model.unet_model is None:
             unet_kwargs["subfolder"] = "unet"
             
        print(f"Carregando UNet limpo de: {unet_path}")
        clean_unet = UNet2DConditionModel.from_pretrained(unet_path, **unet_kwargs)
                
        self.pipe = MVAdapterI2MVSDXLPipeline(
            vae=vae,
            unet=clean_unet,
            scheduler=clean_wrapped_scheduler, 
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2
        )
                
        print("Inicializando adapters no UNet para I2MV...")
        self.pipe.init_custom_adapter(num_views=t2mv_model.num_views)
        
        print("Reutilizando o cond_encoder (fp16) da pipeline T2MV.")
        self.pipe.cond_encoder = t2mv_model.pipe.cond_encoder 
                
        print("Carregando pesos do adapter I2MV no UNet...")
        self.pipe.load_custom_adapter(
            t2mv_model.adapter_path, 
            weight_name="mvadapter_i2mv_sdxl.safetensors"
        )
        
        self.pipe.to(self.device, self.dtype)
        self.pipe.enable_vae_slicing()
        
        print("Pipeline I2MV carregado com sucesso.")

    def run_pipeline(
        self,
        input_image, 
        reference_conditioning_scale,
        text,
        height,
        width,
        num_inference_steps,
        guidance_scale,
        seed,
        negative_prompt,
        azimuth_deg=None,
    ):
        
        pipe = self.pipe
        device = self.device
        
        if azimuth_deg is None:
            azimuth_deg = [180, 270, 315] 
        
        num_views_to_generate = len(azimuth_deg)

        adjusted_azimuth = [x - 90 for x in azimuth_deg]

        cameras = get_orthogonal_camera(
            elevation_deg=[0] * num_views_to_generate,
            distance=[1.8] * num_views_to_generate,
            left=-0.55, right=0.55, bottom=-0.55, top=0.55,
            azimuth_deg=adjusted_azimuth,
            device=device,
        )

        plucker_embeds = get_plucker_embeds_from_cameras_ortho(
            cameras.c2w, [1.1] * num_views_to_generate, width
        )
        control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

        pipe_kwargs = {"max_sequence_length": 214}
        if seed != -1:
            pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

        images = pipe(
            text,
            reference_image=input_image, 
            reference_conditioning_scale=reference_conditioning_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_views_to_generate, 
            control_image=control_images,
            control_conditioning_scale=1.0,
            negative_prompt=negative_prompt,
            **pipe_kwargs,
        ).images

        return images


if __name__ == "__main__":
    
    print("Iniciando o script de geração T2MV -> I2MV...")
    start_time = time.time()  
 
    prompt = "a cute orange cat"
    negative_prompt = "blurry, low quality, distorted"
    seed = 42
    height = 768
    width = 768
    num_inference_steps = 30
    guidance_scale = 7.5
    
    reference_conditioning_scale = .75
    
    adapter_num_views = 6 
    
    output_dir = "generation_batches_img2img"
    os.makedirs(output_dir, exist_ok=True)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Carregando modelos para o dispositivo: {device} com dtype: {dtype}")
    init_start_time = time.time()
    t2mv_model = MVAdapterT2MV( 
        base_model="Lykon/dreamshaper-xl-1-0",  
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        unet_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler="ddpm",
        num_views=adapter_num_views, 
        device=device,
        dtype=dtype,
    )
    print(f"Modelo T2MV carregado em {(time.time() - init_start_time):.2f} segundos.")
    
    i2mv_model = MVAdapterI2MV(t2mv_model) 
 
    # angles_batch_1 = [0, 45, 90]
    # print(f"\nIniciando Geração - Batch 1 [T2MV] (Ângulos: {angles_batch_1})...")
    batch1_start_time = time.time()
    
    images_batch_1 = t2mv_model.run_pipeline(
        text=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed, 
        negative_prompt=negative_prompt,
        # azimuth_deg=angles_batch_1, 
    )
    
    print(f"Batch 1 gerado em {(time.time() - batch1_start_time):.2f} segundos.")
    
    # input_image_for_batch2 = images_batch_1[0]
    # input_image_for_batch2.save(os.path.join(output_dir, "input_image_for_batch2 (from_angle_0).png"))

    # angles_batch_2 = [180, 270, 315]
    # print(f"\nIniciando Geração - Batch 2 [I2MV] (Ângulos: {angles_batch_2})...")
    # print(f"Usando Imagem de Entrada (ângulo 0) com reference_conditioning_scale: {reference_conditioning_scale}")
    # batch2_start_time = time.time()

    # images_batch_2 = i2mv_model.run_pipeline(
    #     input_image=input_image_for_batch2, 
    #     reference_conditioning_scale=reference_conditioning_scale, 
    #     text=prompt, 
    #     height=height,
    #     width=width,
    #     num_inference_steps=num_inference_steps,
    #     guidance_scale=guidance_scale,
    #     seed=seed, 
    #     negative_prompt=negative_prompt,
    #     azimuth_deg=angles_batch_2, 
    # )
    
    # print(f"Batch 2 gerado em {(time.time() - batch2_start_time):.2f} segundos.")

    # print(f"\nSalvando imagens e grades no diretório: '{output_dir}'")
    
    for i, img in enumerate(images_batch_1):
        img.save(os.path.join(output_dir, f"batch1_T2MV.png"))
        
    # for i, img in enumerate(images_batch_2):
    #     img.save(os.path.join(output_dir, f"batch2_I2MV_angle_{angles_batch_2[i]}.png"))

    grid_batch_1 = make_image_grid(images_batch_1, rows=2, cols=3)
    grid_batch_1.save(os.path.join(output_dir, "comparison_grid_batch1_T2MV.png"))
    
    # grid_batch_2 = make_image_grid(images_batch_2, rows=1, cols=3)
    # grid_batch_2.save(os.path.join(output_dir, "comparison_grid_batch2_I2MV.png"))

    # all_images = images_batch_1 + images_batch_2
    # grid_all = make_image_grid(all_images, rows=2, cols=3)
    # grid_all.save(os.path.join(output_dir, "comparison_grid_all_6_views (T2MV+I2MV).png"))

    print("Resultados salvos com sucesso.")

    total_time = time.time() - start_time
    print(f"\nTempo total de execução: {total_time:.2f} segundos.")
    print("="*60)