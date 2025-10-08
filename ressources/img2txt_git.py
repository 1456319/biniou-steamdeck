# https://github.com/Woolverine94/biniou
# img2txt_git.py
import gradio as gr
import os
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, VisionEncoderDecoderModel
import torch
from ressources.common import *

device_label_img2txt_git, model_arch = detect_device()
device_img2txt_git = torch.device(device_label_img2txt_git)

# Gestion des modèles
model_path_img2txt_git = "./models/GIT"
os.makedirs(model_path_img2txt_git, exist_ok=True)

model_list_img2txt_git = [
    "microsoft/git-large-coco",
]

@metrics_decoration
def text_img2txt_git(
    modelid_img2txt_git, 
    max_tokens_img2txt_git, 
    min_tokens_img2txt_git, 
    num_beams_img2txt_git, 
    num_beam_groups_img2txt_git, 
    diversity_penalty_img2txt_git, 
    img_img2txt_git, 
    progress_img2txt_git=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Image captioning 👁️ ]: starting module")
    print(f">>>[Image captioning 👁️ ]: generated 1 caption")
    print(f">>>[Image captioning 👁️ ]: leaving module")
    return "dummy caption"
