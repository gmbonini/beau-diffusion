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
        self.outputs = self.pipe.run_multi_image(
            images,
            seed=seed,
            preprocess_image=False,
            # set parameters in gradio (maybe? idk)
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
        return video, filename

    def save_mesh(self, outputs, glb_path, ply_path, simplify=0.95, texture_size=1024):
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=simplify,
            texture_size=texture_size,
        )
        glb.export(glb_path)
        outputs['gaussian'][0].save_ply(ply_path)
        return glb_path, ply_path

    # def save_mesh(self, outputs, filename="multiview_gradio.obj"):
    #     mesh = outputs['mesh'][0]
    #     mesh.export(filename)
    #     return mesh, filename
    #     # GLB files can be extracted from the outputs
    #     glb = postprocessing_utils.to_glb(
    #         outputs['gaussian'][0],
    #         outputs['mesh'][0],
    #         # Optional parameters
    #         simplify=0.95,          # Ratio of triangles to remove in the simplification process
    #         texture_size=1024,  #1024    # Size of the texture used for the GLB
    #     )
    #     glb.export("sample.glb")


# import os
# # os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
# os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
#                                             # 'auto' is faster but will do benchmarking at the beginning.
#                                             # Recommended to set to 'native' if run only once.

# import numpy as np
# import imageio
# from PIL import Image
# from trellis.pipelines import TrellisImageTo3DPipeline
# from trellis.utils import render_utils
# import torch, time
# # Load a pipeline from a model folder or a Hugging Face model hub.
# pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
# pipeline.cuda()

# # Load an image
# t0 = time.perf_counter()
# images = [
#     Image.open("/mnt/DATA/projects/beau/MV-Adapter/mvadapter/outputs/park3/park_0.png"),
#     Image.open("/mnt/DATA/projects/beau/MV-Adapter/mvadapter/outputs/park3/park_1.png"),
#     Image.open("/mnt/DATA/projects/beau/MV-Adapter/mvadapter/outputs/park3/park_2.png"),
#     Image.open("/mnt/DATA/projects/beau/MV-Adapter/mvadapter/outputs/park3/park_3.png"),
#     Image.open("/mnt/DATA/projects/beau/MV-Adapter/mvadapter/outputs/park3/park_4.png"),
#     Image.open("/mnt/DATA/projects/beau/MV-Adapter/mvadapter/outputs/park3/park_5.png")

# ]


# # Run the pipeline
# outputs = pipeline.run_multi_image(
#     images,
#     seed=1,
#     preprocess_image=False,
#     # Optional parameters
#     sparse_structure_sampler_params={
#         "steps": 50,
#         "cfg_strength": 7.5,
#     },
#     slat_sampler_params={
#         "steps": 12,
#         "cfg_strength": 3,
#     },
# )
# dt = time.perf_counter() - t0
# print(f"pipeline.run: {dt:.3f}s")
# # outputs is a dictionary containing generated 3D assets in different formats:
# # - outputs['gaussian']: a list of 3D Gaussians
# # - outputs['radiance_field']: a list of radiance fields
# # - outputs['mesh']: a list of meshes

# video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
# video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
# video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
# imageio.mimsave("sample_multi.mp4", video, fps=30)
