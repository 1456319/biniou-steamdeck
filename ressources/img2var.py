# https://github.com/Woolverine94/biniou
# img2var.py
import gradio as gr
import os
import PIL
import torch
from diffusers import StableDiffusionImageVariationPipeline
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_label_img2var, model_arch = detect_device()
device_img2var = torch.device(device_label_img2var)

# Gestion des modèles
model_path_img2var = "./models/Stable_Diffusion/"
os.makedirs(model_path_img2var, exist_ok=True)

model_list_img2var = [
    "lambdalabs/sd-image-variations-diffusers",
]

# Bouton Cancel
stop_img2var = False

def initiate_stop_img2var() :
    global stop_img2var
    stop_img2var = True

def check_img2var(step, timestep, latents) : 
    global stop_img2var
    if stop_img2var == False :
        return
    elif stop_img2var == True :
        print(">>>[Image variation 🖼️ ]: generation canceled by user")
        stop_img2var = False
        try:
            del ressources.img2var.pipe_img2var
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def image_img2var(
    modelid_img2var, 
    sampler_img2var, 
    img_img2var, 
    num_images_per_prompt_img2var, 
    num_prompt_img2var, 
    guidance_scale_img2var, 
    num_inference_step_img2var, 
    height_img2var, 
    width_img2var, 
    seed_img2var, 
    use_gfpgan_img2var, 
    nsfw_filter, 
    tkme_img2var,    
    progress_img2var=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Image variation 🖼️ ]: starting module")
    print(f">>>[Image variation 🖼️ ]: generated {num_prompt_img2var} batch(es) of {num_images_per_prompt_img2var}")
    print(f">>>[Image variation 🖼️ ]: leaving module")  
    return ["dummy.png"], ["dummy.png"]
