# https://github.com/Woolverine94/biniou
# img2vid.py
import gradio as gr
import os
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, export_to_gif
import numpy as np
import torch
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_label_img2vid, model_arch = detect_device()
device_img2vid = torch.device(device_label_img2vid)

model_path_img2vid = "./models/Stable_Video_Diffusion/"
model_path_safetychecker_img2vid = "./models/Stable_Diffusion/"
os.makedirs(model_path_img2vid, exist_ok=True)

model_list_img2vid = [
    "stabilityai/stable-video-diffusion-img2vid",
    "stabilityai/stable-video-diffusion-img2vid-xt",
]

# Bouton Cancel
stop_img2vid = False

def initiate_stop_img2vid() :
    global stop_img2vid
    stop_img2vid = True

def check_img2vid(pipe, step_index, timestep, callback_kwargs):
    global stop_img2vid
    if stop_img2vid == False :
        return callback_kwargs
    elif stop_img2vid == True :
        print(">>>[Text2Video-Zero 📼 ]: generation canceled by user")
        stop_img2vid = False
        try:
            del ressources.img2vid.pipe_img2vid
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def video_img2vid(
    modelid_img2vid,
    num_inference_steps_img2vid,
    sampler_img2vid,
    min_guidance_scale_img2vid,
    max_guidance_scale_img2vid,
    seed_img2vid,
    num_frames_img2vid,
    num_fps_img2vid,
    decode_chunk_size_img2vid,
    width_img2vid,
    height_img2vid,
    num_prompt_img2vid,
    num_videos_per_prompt_img2vid,
    motion_bucket_id_img2vid,
    noise_aug_strength_img2vid,
    nsfw_filter,
    img_img2vid,
    output_type_img2vid,
    use_gfpgan_img2vid,
    tkme_img2vid,
    progress_img2vid=gr.Progress(track_tqdm=True)
    ):
    
    print(">>>[Stable Video Diffusion 📼 ]: starting module")
    print(f">>>[Stable Video Diffusion 📼 ]: generated {num_prompt_img2vid} batch(es) of {num_videos_per_prompt_img2vid}")
    print(f">>>[Stable Video Diffusion 📼 ]: leaving module")
    return "dummy.mp4"
