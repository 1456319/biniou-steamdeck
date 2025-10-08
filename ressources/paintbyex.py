# https://github.com/Woolverine94/biniou
# paintbyex.py
import gradio as gr
import os
import PIL
import torch
from diffusers import PaintByExamplePipeline
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_label_paintbyex, model_arch = detect_device()
device_paintbyex = torch.device(device_label_paintbyex)

# Gestion des modèles
model_path_paintbyex = "./models/Paint_by_example/"
model_path_safety_checker = "./models/Stable_Diffusion/"
os.makedirs(model_path_paintbyex, exist_ok=True)
os.makedirs(model_path_safety_checker, exist_ok=True)
model_list_paintbyex = []

for filename in os.listdir(model_path_paintbyex):
    f = os.path.join(model_path_paintbyex, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_paintbyex.append(f)

model_list_paintbyex_builtin = [
    "Fantasy-Studio/Paint-by-Example",
]

for k in range(len(model_list_paintbyex_builtin)):
    model_list_paintbyex.append(model_list_paintbyex_builtin[k])

# Bouton Cancel
stop_paintbyex = False

def initiate_stop_paintbyex() :
    global stop_paintbyex
    stop_paintbyex = True

def check_paintbyex(step, timestep, latents) : 
    global stop_paintbyex
    if stop_paintbyex == False :
        return
    elif stop_paintbyex == True :
        print(">>>[Paint by example 🖌️ ]: generation canceled by user")
        stop_paintbyex = False
        try:
            del ressources.paintbyex.pipe_paintbyex
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def image_paintbyex(
    modelid_paintbyex, 
    sampler_paintbyex, 
    img_paintbyex, 
    rotation_img_paintbyex, 
    example_img_paintbyex, 
    num_images_per_prompt_paintbyex, 
    num_prompt_paintbyex, 
    guidance_scale_paintbyex,
    num_inference_step_paintbyex, 
    height_paintbyex, 
    width_paintbyex, 
    seed_paintbyex, 
    use_gfpgan_paintbyex, 
    nsfw_filter, 
    tkme_paintbyex,
    progress_paintbyex=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Paint by example 🖌️ ]: starting module")
    print(f">>>[Paint by example 🖌️ ]: generated {num_prompt_paintbyex} batch(es) of {num_images_per_prompt_paintbyex}")
    print(f">>>[Paint by example 🖌️ ]: leaving module")
    return ["dummy.png"], ["dummy.png"]
