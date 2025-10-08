# https://github.com/Woolverine94/biniou
# animatediff_lcm.py
import gradio as gr
import os
import imageio
from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers.utils import export_to_video, export_to_gif
import numpy as np
from compel import Compel, ReturnedEmbeddingsType
import torch
import random
from ressources.common import *
from ressources.gfpgan import *
from huggingface_hub import snapshot_download, hf_hub_download
from safetensors.torch import load_file
import tomesd
import shutil

device_label_animatediff_lcm, model_arch = detect_device()
device_animatediff_lcm = torch.device(device_label_animatediff_lcm)

model_path_animatediff_lcm = "./models/Stable_Diffusion/"
os.makedirs(model_path_animatediff_lcm, exist_ok=True)

adapter_path_animatediff_lcm = "./models/AnimateLCM/"
os.makedirs(model_path_animatediff_lcm, exist_ok=True)

lora_path_animatediff_lcm = "./models/AnimateLCM/LoRA"
os.makedirs(model_path_animatediff_lcm, exist_ok=True)


model_list_animatediff_lcm = [
    "emilianJR/epiCRealism",
    "SG161222/Realistic_Vision_V3.0_VAE",
#    "stabilityai/sdxl-turbo",
#    "dataautogpt3/OpenDalleV1.1",
    "digiplay/AbsoluteReality_v1.8.1",
#    "segmind/Segmind-Vega",
#    "segmind/SSD-1B",
    "gsdf/Counterfeit-V2.5",
#    "ckpt/anything-v4.5-vae-swapped",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion",
]

model_list_adapters_animatediff_lcm = {
    "wangfuyun/AnimateLCM":("AnimateLCM_sd15_t2v_lora.safetensors", 0.8),
    "ByteDance/AnimateDiff-Lightning":("animatediff_lightning_4step_diffusers.safetensors", 1.0),
}

# Bouton Cancel
stop_animatediff_lcm = False

def initiate_stop_animatediff_lcm() :
    global stop_animatediff_lcm
    stop_animatediff_lcm = True

def check_animatediff_lcm(pipe, step_index, timestep, callback_kwargs) :
    global stop_animatediff_lcm
    if stop_animatediff_lcm == True :
        print(">>>[AnimateLCM 📼 ]: generation canceled by user")
        stop_animatediff_lcm = False
        pipe._interrupt = True
    return callback_kwargs

@metrics_decoration
def video_animatediff_lcm(
    modelid_animatediff_lcm,
    modelid_adapters_animatediff_lcm,
    num_inference_step_animatediff_lcm,
    sampler_animatediff_lcm,
    guidance_scale_animatediff_lcm,
    seed_animatediff_lcm,
    num_frames_animatediff_lcm,
    num_fps_animatediff_lcm,
    height_animatediff_lcm,
    width_animatediff_lcm,
    num_videos_per_prompt_animatediff_lcm,
    num_prompt_animatediff_lcm,
    prompt_animatediff_lcm,
    negative_prompt_animatediff_lcm,
    output_type_animatediff_lcm,
    nsfw_filter,
    use_gfpgan_animatediff_lcm,
    tkme_animatediff_lcm,
    clipskip_animatediff_lcm,
    progress_animatediff_lcm=gr.Progress(track_tqdm=True)
    ):

    print(">>>[AnimateLCM 📼 ]: starting module")
    print(f">>>[AnimateLCM 📼 ]: generated {num_prompt_animatediff_lcm} batch(es) of {num_videos_per_prompt_animatediff_lcm}")
    print(f">>>[AnimateLCM 📼 ]: leaving module")
    return "dummy.mp4"
