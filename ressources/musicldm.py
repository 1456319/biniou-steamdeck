# https://github.com/Woolverine94/biniou
# musicldm.py
import os
import gradio as gr
from diffusers import MusicLDMPipeline
import torch
import scipy
import random
from ressources.common import *

device_label_musicldm, model_arch = detect_device()
device_musicldm = torch.device(device_label_musicldm)

model_path_musicldm = "./models/MusicLDM/"
os.makedirs(model_path_musicldm, exist_ok=True)

model_list_musicldm = [
    "ucsd-reach/musicldm",
    "sanchit-gandhi/musicldm-full",
]

# Bouton Cancel
stop_musicldm = False

def initiate_stop_musicldm() :
    global stop_musicldm
    stop_musicldm = True

def check_musicldm(step, timestep, latents) : 
    global stop_musicldm
    if stop_musicldm == False :
        return
    elif stop_musicldm == True :
        print(">>>[MusicLDM 🎶 ]: generation canceled by user")
        stop_musicldm = False
        try:
            del ressources.musicldm.pipe_musicldm
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def music_musicldm(
    modelid_musicldm, 
    sampler_musicldm, 
    prompt_musicldm, 
    negative_prompt_musicldm, 
    num_audio_per_prompt_musicldm, 
    num_prompt_musicldm, 
    guidance_scale_musicldm, 
    num_inference_step_musicldm, 
    audio_length_musicldm,
    seed_musicldm,    
    progress_musicldm=gr.Progress(track_tqdm=True)
    ):

    print(">>>[MusicLDM 🎶 ]: starting module")
    print(f">>>[MusicLDM 🎶 ]: generated {num_prompt_musicldm} batch(es) of {num_audio_per_prompt_musicldm} audio")
    print(f">>>[MusicLDM 🎶 ]: leaving module")
    return "dummy.wav"
