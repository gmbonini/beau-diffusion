import sys
import numpy as np
import imageio
import os 

TRELLIS_PATH = "/workspace/beau/TRELLIS"
if TRELLIS_PATH not in sys.path:
    sys.path.append(TRELLIS_PATH)

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.utils import render_utils, postprocessing_utils

os.environ['ATTN_BACKEND'] = 'xformers' 

class TrellisMV2M:
    def __init__(
        self,
        model = "microsoft/TRELLIS-image-large",
        device: str = "cuda",
        ):
        self.model = model
        self.device = device
        self.pipe = None
        self.outputs = None

    def prepare_pipeline(self):
        pipe = TrellisImageTo3DPipeline.from_pretrained(self.model)
        pipe.cuda()
        
        self.pipe = pipe
        return pipe

    def run_pipeline(self, images, seed):
        print('Images list:', images)
        self.outputs = self.pipe.run_multi_image(
            images,
            seed=seed,
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": 50,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": 3,
            },
        )
        return self.outputs

    def save_video(self, outputs, filename="multiview_gradio.mp4"):
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        imageio.mimsave(filename, video, fps=30)
        
        middle_frame_np = video[len(video) // 2]

        return video, filename, middle_frame_np

    def save_mesh(self, outputs, glb_path, ply_path, simplify=0.80, texture_size=1024):
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=simplify,
            texture_size=texture_size,
        )
        glb.export(glb_path)
        outputs['gaussian'][0].save_ply(ply_path)
        return glb_path, ply_path