# https://github.com/Woolverine94/biniou
# Audiogen.py
import os
import gradio as gr
import torch
import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import random
from ressources.common import *

device_label_audiogen, model_arch = detect_device()
device_audiogen = torch.device(device_label_audiogen)

model_path_audiogen = "./models/Audiocraft/"
os.makedirs(model_path_audiogen, exist_ok=True)

modellist_audiogen = [
    "facebook/audiogen-medium",
    "AkhilTolani/audiogen-v2",
]

# Bouton Cancel
stop_audiogen = False

def initiate_stop_audiogen() :
    global stop_audiogen
    stop_audiogen = True

def check_audiogen(generated_tokens, total_tokens) : 
    global stop_audiogen
    if stop_audiogen == False :
        return
    elif stop_audiogen == True :
        print(">>>[AudioGen 🔊 ]: generation canceled by user")
        stop_audiogen = False
        try:
            del ressources.audiogen.pipe_audiogen
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def music_audiogen(
    prompt_audiogen, 
    model_audiogen, 
    duration_audiogen, 
    num_batch_audiogen, 
    temperature_audiogen, 
    top_k_audiogen, 
    top_p_audiogen, 
    use_sampling_audiogen, 
    cfg_coef_audiogen, 
    progress_audiogen=gr.Progress(track_tqdm=True)
    ):

    print(">>>[AudioGen 🔊 ]: starting module")
    print(f">>>[AudioGen 🔊 ]: generated {num_batch_audiogen} batch(es) of 1")
    print(f">>>[AudioGen 🔊 ]: leaving module")
    return "dummy.wav"
