# https://github.com/Woolverine94/biniou
# img2shape.py
import gradio as gr
import os
import PIL
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_gif, export_to_ply
import torch
import random
import trimesh
import numpy as np
from ressources.common import *

device_label_img2shape, model_arch = detect_device()
device_img2shape = torch.device(device_label_img2shape)

# Gestion des modèles
model_path_img2shape = "./models/Shap-E/"
model_path_img2shape_safetychecker = "./models/Stable_Diffusion/"
os.makedirs(model_path_img2shape, exist_ok=True)
model_list_img2shape = []

for filename in os.listdir(model_path_img2shape):
    f = os.path.join(model_path_img2shape, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_img2shape.append(f)

model_list_img2shape_builtin = [
    "openai/shap-e-img2img", 
]

for k in range(len(model_list_img2shape_builtin)):
    model_list_img2shape.append(model_list_img2shape_builtin[k])

# Bouton Cancel
stop_img2shape = False

def initiate_stop_img2shape() :
    global stop_img2shape
    stop_img2shape = True

def check_img2shape(step, timestep, latents) :
    global stop_img2shape
    if stop_img2shape == False :
#        result_preview = preview_image(step, timestep, latents, pipe_img2shape)
        return
    elif stop_img2shape == True :
        stop_img2shape = False
        try:
            del ressources.img2shape.pipe_img2shape
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def image_img2shape(
    modelid_img2shape, 
    sampler_img2shape,  
    img_img2shape, 
    num_images_per_prompt_img2shape, 
    num_prompt_img2shape, 
    guidance_scale_img2shape, 
    num_inference_step_img2shape, 
    frame_size_img2shape, 
    seed_img2shape, 
    output_type_img2shape,     
    nsfw_filter, 
    progress_img2shape=gr.Progress(track_tqdm=True)
    ):
    
    print(">>>[Shap-E img2shape 🧊 ]: starting module") 
    print(f">>>[Shap-E img2shape 🧊 ]: generated {num_prompt_img2shape} batch(es) of {num_images_per_prompt_img2shape}")
    print(f">>>[Shap-E img2shape 🧊 ]: leaving module")
    if output_type_img2shape=="gif":
        return "dummy.gif", "dummy.gif"
    else:
        return "dummy.glb", "dummy.glb"
