# https://github.com/Woolverine94/biniou
# outpaint.py
import gradio as gr
import os
import PIL
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
from compel import Compel, ReturnedEmbeddingsType
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd
from diffusers.schedulers import AysSchedules

device_label_outpaint, model_arch = detect_device()
device_outpaint = torch.device(device_label_outpaint)

# Gestion des modèles
model_path_outpaint = "./models/inpaint/"
model_path_outpaint_safety_checker = "./models/Stable_Diffusion/"
os.makedirs(model_path_outpaint, exist_ok=True)
os.makedirs(model_path_outpaint_safety_checker, exist_ok=True)
model_list_outpaint = []

for filename in os.listdir(model_path_outpaint):
    f = os.path.join(model_path_outpaint, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_outpaint.append(f)

model_list_outpaint_builtin = [
    "Uminosachi/realisticVisionV51_v51VAE-inpainting",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
#    "runwayml/stable-diffusion-inpainting",
    "Lykon/dreamshaper-8-inpainting",
    "Sanster/anything-4.0-inpainting",
    "kpsss34/inpaintingXL",
]

for k in range(len(model_list_outpaint_builtin)):
    model_list_outpaint.append(model_list_outpaint_builtin[k])

# Bouton Cancel
stop_outpaint = False

def initiate_stop_outpaint() :
    global stop_outpaint
    stop_outpaint = True

def check_outpaint(pipe, step_index, timestep, callback_kwargs) : 
    global stop_outpaint
    if stop_outpaint == True :
        print(">>>[outpaint 🖌️ ]: generation canceled by user")
        stop_outpaint = False
        pipe._interrupt = True
    return callback_kwargs

def prepare_outpaint(img_outpaint, top, bottom, left, right) :
    image = np.array(img_outpaint)
    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype = np.uint8)
    top = int(top)
    bottom = int(bottom)
    left = int(left)
    right = int(right)
    image = cv2.copyMakeBorder(
        image, 
        top, 
        bottom, 
        left, 
        right, 
        cv2.BORDER_CONSTANT, 
        None, 
        [255, 255, 255]
    )
    mask = cv2.copyMakeBorder(
        mask, 
        top, 
        bottom, 
        left, 
        right, 
        cv2.BORDER_CONSTANT, 
        None, 
        [255, 255, 255]
    )
    return image, image, mask, mask

@metrics_decoration
def image_outpaint(
    modelid_outpaint, 
    sampler_outpaint, 
    img_outpaint, 
    mask_outpaint, 
    rotation_img_outpaint, 
    prompt_outpaint, 
    negative_prompt_outpaint, 
    num_images_per_prompt_outpaint, 
    num_prompt_outpaint, 
    guidance_scale_outpaint,
    denoising_strength_outpaint, 
    num_inference_step_outpaint, 
    height_outpaint, 
    width_outpaint, 
    seed_outpaint, 
    use_gfpgan_outpaint, 
    nsfw_filter, 
    tkme_outpaint,
    clipskip_outpaint,
    use_ays_outpaint,
    progress_outpaint=gr.Progress(track_tqdm=True)
    ):

    print(">>>[outpaint 🖌️ ]: starting module")
    print(f">>>[outpaint 🖌️ ]: generated {num_prompt_outpaint} batch(es) of {num_images_per_prompt_outpaint}")
    print(f">>>[outpaint 🖌️ ]: leaving module")
    return ["dummy.png"], ["dummy.png"]
