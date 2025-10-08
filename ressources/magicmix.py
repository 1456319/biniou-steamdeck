# https://github.com/Woolverine94/biniou
# magicmix.py
import gradio as gr
import os
import torch
from diffusers import DiffusionPipeline
import random
from ressources.gfpgan import *
import tomesd

# device_magicmix = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_label_magicmix, model_arch = detect_device()
device_magicmix = torch.device(device_label_magicmix)

# Gestion des modèles
model_path_magicmix = "./models/Stable_Diffusion/"
os.makedirs(model_path_magicmix, exist_ok=True)
model_list_magicmix = []

for filename in os.listdir(model_path_magicmix):
    f = os.path.join(model_path_magicmix, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_magicmix.append(f)

model_list_magicmix_builtin = [
    "SG161222/Realistic_Vision_V3.0_VAE",
#    "ckpt/anything-v4.5-vae-swapped",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion", 
]

for k in range(len(model_list_magicmix_builtin)):
    model_list_magicmix.append(model_list_magicmix_builtin[k])

# Bouton Cancel
stop_magicmix = False

def initiate_stop_magicmix() :
    global stop_magicmix
    stop_magicmix = True

def check_magicmix(step, timestep, latents) :
    global stop_magicmix
    if stop_magicmix == False :
        return
    elif stop_magicmix == True :
        print(">>>[MagicMix 🖼️ ]: generation canceled by user")
        stop_magicmix = False
        try:
            del ressources.magicmix.pipe_magicmix
        except NameError as e:
            raise Exception("Interrupting ...")
            return "Canceled ..."
    return

@metrics_decoration
def image_magicmix(
    modelid_magicmix, 
    sampler_magicmix, 
    num_inference_step_magicmix,
    guidance_scale_magicmix,
    kmin_magicmix,
    kmax_magicmix,
    num_prompt_magicmix,
    seed_magicmix,
    img_magicmix,
    prompt_magicmix,
    mix_factor_magicmix,
    use_gfpgan_magicmix, 
    nsfw_filter, 
    tkme_magicmix,
    progress_magicmix=gr.Progress(track_tqdm=True)
    ):

    print(">>>[MagicMix 🖼️ ]: starting module")
    print(f">>>[MagicMix 🖼️ ]: generated {num_prompt_magicmix} batch(es) of 1")
    print(f">>>[MagicMix 🖼️ ]: leaving module")
    return ["dummy.png"], ["dummy.png"]
