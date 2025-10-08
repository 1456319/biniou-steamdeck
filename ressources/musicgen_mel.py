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

device_label_musicgen_mel, model_arch = detect_device()
device_musicgen_mel = torch.device(device_label_musicgen_mel)

model_path_musicgen_mel = "./models/Audiocraft/"
os.makedirs(model_path_musicgen_mel, exist_ok=True)

modellist_musicgen_mel = [
    "facebook/musicgen-stereo-melody",
    "facebook/musicgen-melody",
    "facebook/musicgen-stereo-melody-large",
    "facebook/musicgen-melody-large",
    "nateraw/musicgen-songstarter-v0.2",
    "facebook/musicgen-style",
]

# Bouton Cancel
stop_musicgen_mel = False

def initiate_stop_musicgen_mel() :
    global stop_musicgen_mel
    stop_musicgen_mel = True

def check_musicgen_mel(generated_tokens, total_tokens) : 
    global stop_musicgen_mel
    if stop_musicgen_mel == False :
        return
    elif stop_musicgen_mel == True :
        print(">>>[MusicGen Melody 🎶 ]: generation canceled by user")
        stop_musicgen_mel = False
        try:
            del ressources.musicgen.pipe_musicgen_mel
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def music_musicgen_mel(
    prompt_musicgen_mel, 
    model_musicgen_mel, 
    duration_musicgen_mel, 
    num_batch_musicgen_mel, 
    temperature_musicgen_mel, 
    top_k_musicgen_mel, 
    top_p_musicgen_mel, 
    use_sampling_musicgen_mel, 
    cfg_coef_musicgen_mel, 
    source_audio_musicgen_mel,
    source_type_musicgen_mel,
    progress_musicgen_mel=gr.Progress(track_tqdm=True)
    ):

    print(">>>[MusicGen Melody 🎶 ]: starting module")
    print(f">>>[MusicGen Melody 🎶 ]: generated {num_batch_musicgen_mel} batch(es) of 1")
    print(f">>>[MusicGen Melody 🎶 ]: leaving module")
    return "dummy.wav"
