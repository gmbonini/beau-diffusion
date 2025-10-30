# https://app.clickup.com/t/86acvf0jj

import argparse
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
        self.dtype = dtype

        # Preparar componentes compartilhados
        self.vae, self.scheduler_class_enum, self.lora_model_list, self.adapter_name_list = self._prepare_components(
            vae_model, scheduler, lora_model, device, dtype
        )
        
        # Criar pipeline inicial
        self.pipe = self.load_pipeline()
 
    def _prepare_components(self, vae_model, scheduler, lora_model, device, dtype):
        vae = None
        if vae_model is not None:
            vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype)
        
        scheduler_class_enum = None
        if scheduler == "ddpm":
            scheduler_class_enum = DDPMScheduler
        elif scheduler == "lcm":
            scheduler_class_enum = LCMScheduler
        
        # Processar LoRA models
        lora_model_list = []
        adapter_name_list = []
        if lora_model is not None:
            lora_model_list = lora_model.split(",")
            for lora_model_ in lora_model_list:
                model_, name_ = lora_model_.strip().rsplit("/", 1)
                adapter_name = name_.split(".")[0]
                adapter_name_list.append(adapter_name)
        
        return vae, scheduler_class_enum, lora_model_list, adapter_name_list

    def load_pipeline(self):
        """Cria um pipeline limpo com UNet e scheduler novos"""
        print("Carregando pipeline limpo...")
        
        # Carregar UNet limpo
        unet_path = self.unet_model if self.unet_model else self.base_model
        unet_kwargs = {"torch_dtype": self.dtype}
        if self.unet_model is None:
            unet_kwargs["subfolder"] = "unet"
        
        print(f"Carregando UNet limpo de: {unet_path}")
        clean_unet = UNet2DConditionModel.from_pretrained(unet_path, **unet_kwargs)
        
        # Preparar kwargs do pipeline
        pipe_kwargs = {"unet": clean_unet}
        if self.vae is not None:
            pipe_kwargs["vae"] = self.vae
        
        # Criar pipeline
        pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(
            self.base_model, 
            torch_dtype=self.dtype, 
            **pipe_kwargs
        )
        
        # Criar scheduler limpo
        base_scheduler_config = pipe.scheduler.config
        clean_base_scheduler = self.scheduler_class_enum.from_config(base_scheduler_config) if self.scheduler_class_enum else pipe.scheduler
        
        # Envolver no ShiftSNRScheduler
        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            clean_base_scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=self.scheduler_class_enum,
        )
        
        # Inicializar e carregar adapters
        print("Inicializando adapters no UNet...")
        pipe.init_custom_adapter(num_views=self.num_views)
        
        print("Carregando pesos do adapter T2MV...")
        pipe.load_custom_adapter(
            self.adapter_path, 
            weight_name="mvadapter_t2mv_sdxl.safetensors"
        )
        
        # Mover para dispositivo
        pipe.to(device=self.device, dtype=self.dtype)
        pipe.cond_encoder.to(device=self.device, dtype=self.dtype)
        
        # Carregar LoRAs se existirem
        if len(self.lora_model_list) > 0:
            for lora_model_, adapter_name in zip(self.lora_model_list, self.adapter_name_list):
                model_, name_ = lora_model_.strip().rsplit("/", 1)
                pipe.load_lora_weights(model_, weight_name=name_, adapter_name=adapter_name)
        
        pipe.enable_vae_slicing()
        
        print("Pipeline carregado com sucesso.")
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
        lora_scale=["1.0"],
        azimuth_deg=None,
    ):
        pipe = self.pipe
        device = self.device
        
        if azimuth_deg is None:
            azimuth_deg = [0, 45, 90, 180, 270, 315]
        
        num_views_to_generate = len(azimuth_deg)
 
        # Configurar LoRAs
        if len(self.adapter_name_list) > 0:
            if len(lora_scale) == 1:
                lora_scale = [lora_scale[0]] * len(self.adapter_name_list)
            else:
                assert len(lora_scale) == len(
                    self.adapter_name_list
                ), "Number of lora scales must match number of adapters"
            lora_scale = [float(s) for s in lora_scale]
            pipe.set_adapters(self.adapter_name_list, adapter_weights=lora_scale)
            print(f"Loaded {len(self.adapter_name_list)} adapters with scales {lora_scale}")
 
        # Ajustar azimuth (subtrair 90 graus como no I2MV)
        adjusted_azimuth = [x - 90 for x in azimuth_deg]
        
        cameras = get_orthogonal_camera(
            elevation_deg=[0] * num_views_to_generate,
            distance=[1.8] * num_views_to_generate,
            left=-0.55,
            right=0.55,
            bottom=-0.55,
            top=0.55,
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
    
    print("Iniciando o script de geração...")
    start_time = time.time()  
 
    prompt = "a cute orange cat"
    negative_prompt = "blurry, low quality, distorted"
    seed = 42
    height = 768
    width = 768
    num_inference_steps = 30
    guidance_scale = 7.5
    
    adapter_num_views = 6 
    
    output_dir = "generation_batches"
    os.makedirs(output_dir, exist_ok=True)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Carregando modelos para o dispositivo: {device} com dtype: {dtype}")
    init_start_time = time.time()
    model = MVAdapterT2MV(
        base_model="Lykon/dreamshaper-xl-1-0",  
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        unet_model=None,
        lora_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler="ddpm",
        num_views=adapter_num_views, 
        device=device,
        dtype=dtype,
    )
    print(f"Modelos carregados em {(time.time() - init_start_time):.2f} segundos.")
 
    angles_batch_1 = [0, 45, 90]
    print(f"\nIniciando Geração - Batch 1 (Ângulos: {angles_batch_1})...")
    batch1_start_time = time.time()
    
    images_batch_1 = model.run_pipeline(
        text=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed, 
        negative_prompt=negative_prompt,
        azimuth_deg=angles_batch_1, 
    )
    
    print(f"Batch 1 gerado em {(time.time() - batch1_start_time):.2f} segundos.")
 
    angles_batch_2 = [180, 270, 315]
    print(f"\nIniciando Geração - Batch 2 (Ângulos: {angles_batch_2})...")
    batch2_start_time = time.time()

    images_batch_2 = model.run_pipeline(
        text=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed, 
        negative_prompt=negative_prompt,
        azimuth_deg=angles_batch_2, 
    )
    
    print(f"Batch 2 gerado em {(time.time() - batch2_start_time):.2f} segundos.")

    print(f"\nSalvando imagens e grades no diretório: '{output_dir}'")
    
    # Salvar imagens individuais
    for i, img in enumerate(images_batch_1):
        img.save(os.path.join(output_dir, f"batch1_angle_{angles_batch_1[i]}.png"))
        
    for i, img in enumerate(images_batch_2):
        img.save(os.path.join(output_dir, f"batch2_angle_{angles_batch_2[i]}.png"))

    grid_batch_1 = make_image_grid(images_batch_1, rows=1, cols=3)
    grid_batch_1.save(os.path.join(output_dir, "comparison_grid_batch1.png"))
    
    grid_batch_2 = make_image_grid(images_batch_2, rows=1, cols=3)
    grid_batch_2.save(os.path.join(output_dir, "comparison_grid_batch2.png"))

    all_images = images_batch_1 + images_batch_2
    grid_all = make_image_grid(all_images, rows=2, cols=3)
    grid_all.save(os.path.join(output_dir, "comparison_grid_all_6_views.png"))

    print("Resultados salvos com sucesso.")

    total_time = time.time() - start_time
    print(f"\nTempo total de execução: {total_time:.2f} segundos.")
    print("="*60)