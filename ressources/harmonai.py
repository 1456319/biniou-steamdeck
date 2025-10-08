# https://github.com/Woolverine94/biniou
# Harmonai.py
import gradio as gr
import os
from diffusers import DiffusionPipeline
import scipy.io.wavfile
import torch
import random
from ressources.common import *

device_label_harmonai, model_arch = detect_device()
device_harmonai = torch.device(device_label_harmonai)

model_path_harmonai = "./models/harmonai/"
os.makedirs(model_path_harmonai, exist_ok=True)

model_list_harmonai = []

for filename in os.listdir(model_path_harmonai):
    f = os.path.join(model_path_harmonai, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors') or filename.endswith('.bin')):
        model_list_harmonai.append(f)

model_list_harmonai_builtin = [
    "harmonai/glitch-440k",
    "harmonai/honk-140k",
    "harmonai/jmann-small-190k",
    "harmonai/jmann-large-580k",
    "harmonai/maestro-150k",
    "harmonai/unlocked-250k",
]

for k in range(len(model_list_harmonai_builtin)):
    model_list_harmonai.append(model_list_harmonai_builtin[k])

@metrics_decoration
def music_harmonai(
    length_harmonai, 
    model_harmonai, 
    steps_harmonai, 
    seed_harmonai, 
    batch_size_harmonai, 
    batch_repeat_harmonai, 
    progress_harmonai=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Harmonai 🔊 ]: starting module")
    print(f">>>[Harmonai 🔊 ]: generated {batch_repeat_harmonai} batch(es) of {batch_size_harmonai}")
    print(f">>>[Harmonai 🔊 ]: leaving module")
    return "dummy.wav"
