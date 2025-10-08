# https://github.com/Woolverine94/biniou
# inpaint.py
import gradio as gr
import os
import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
from compel import Compel, ReturnedEmbeddingsType
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd
from diffusers.schedulers import AysSchedules

device_label_inpaint, model_arch = detect_device()
device_inpaint = torch.device(device_label_inpaint)

# Gestion des modèles
model_path_inpaint = "./models/inpaint/"
model_path_inpaint_safety_checker = "./models/Stable_Diffusion/"
os.makedirs(model_path_inpaint, exist_ok=True)
os.makedirs(model_path_inpaint_safety_checker, exist_ok=True)
model_list_inpaint = []

for filename in os.listdir(model_path_inpaint):
    f = os.path.join(model_path_inpaint, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_inpaint.append(f)

model_list_inpaint_builtin = [
    "Uminosachi/realisticVisionV51_v51VAE-inpainting",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
#    "runwayml/stable-diffusion-inpainting",
    "Lykon/dreamshaper-8-inpainting",
    "Sanster/anything-4.0-inpainting",
    "kpsss34/inpaintingXL",
]

for k in range(len(model_list_inpaint_builtin)):
    model_list_inpaint.append(model_list_inpaint_builtin[k])

# Bouton Cancel
stop_inpaint = False

def initiate_stop_inpaint() :
    global stop_inpaint
    stop_inpaint = True

def check_inpaint(pipe, step_index, timestep, callback_kwargs):
    global stop_inpaint
    if stop_inpaint == True:
        print(">>>[inpaint 🖌️ ]: generation canceled by user")
        stop_inpaint = False
        pipe._interrupt = True
    return callback_kwargs

@metrics_decoration
def image_inpaint(
    modelid_inpaint, 
    sampler_inpaint, 
    img_inpaint, 
    rotation_img_inpaint, 
    prompt_inpaint, 
    negative_prompt_inpaint, 
    num_images_per_prompt_inpaint, 
    num_prompt_inpaint, 
    guidance_scale_inpaint,
    denoising_strength_inpaint, 
    num_inference_step_inpaint, 
    height_inpaint, 
    width_inpaint, 
    seed_inpaint, 
    use_gfpgan_inpaint, 
    nsfw_filter, 
    tkme_inpaint,
    clipskip_inpaint,
    use_ays_inpaint,
    progress_inpaint=gr.Progress(track_tqdm=True)
    ):

    print(">>>[inpaint 🖌️ ]: starting module")
    print(f">>>[inpaint 🖌️ ]: generated {num_prompt_inpaint} batch(es) of {num_images_per_prompt_inpaint}")
    print(f">>>[inpaint 🖌️ ]: leaving module")
    return ["dummy.png"], ["dummy.png"]
