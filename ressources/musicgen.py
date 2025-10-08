# https://github.com/Woolverine94/biniou
# Musicgen.py
import os
import gradio as gr
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import random
from ressources.common import *

device_label_musicgen, model_arch = detect_device()
device_musicgen = torch.device(device_label_musicgen)

model_path_musicgen = "./models/Audiocraft/"
os.makedirs(model_path_musicgen, exist_ok=True)

modellist_musicgen = [
    "facebook/musicgen-stereo-small",
    "facebook/musicgen-small",
    "facebook/musicgen-stereo-medium",
    "facebook/musicgen-medium",
    "facebook/musicgen-stereo-large",
    "facebook/musicgen-large",
    "pharoAIsanders420/musicgen-stereo-dub",
]

# Bouton Cancel
stop_musicgen = False

def initiate_stop_musicgen() :
    global stop_musicgen
    stop_musicgen = True

def check_musicgen(generated_tokens, total_tokens) : 
    global stop_musicgen
    if stop_musicgen == False :
        return
    elif stop_musicgen == True :
        print(">>>[MusicGen 🎶 ]: generation canceled by user")
        stop_musicgen = False
        try:
            del ressources.musicgen.pipe_musicgen
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def music_musicgen(
    prompt_musicgen, 
    model_musicgen, 
    duration_musicgen, 
    num_batch_musicgen, 
    temperature_musicgen, 
    top_k_musicgen, 
    top_p_musicgen, 
    use_sampling_musicgen, 
    cfg_coef_musicgen, 
    progress_musicgen=gr.Progress(track_tqdm=True)
    ):

    print(">>>[MusicGen 🎶 ]: starting module")
    print(f">>>[MusicGen 🎶 ]: generated {num_batch_musicgen} batch(es) of 1")
    print(f">>>[MusicGen 🎶 ]: leaving module")
    return "dummy.wav"
