# https://github.com/Woolverine94/biniou
# r_esrgan.py
import gradio as gr
import os
import PIL
import time
import torch
import numpy as np
from RealESRGAN import RealESRGAN
# from realesrgan import RealESRGANModel as RE
from ressources.common import *
from ressources.gfpgan import *

device_resrgan = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path_resrgan = "./models/Real_ESRGAN/"
os.makedirs(model_path_resrgan, exist_ok=True)

model_list_resrgan = [
    "RealESRGAN_x2.pth",
    "RealESRGAN_x4.pth",
    "RealESRGAN_x8.pth",
]

@metrics_decoration
def image_resrgan(
    modelid_resrgan, 
    scale_resrgan, 
    img_resrgan, 
    use_gfpgan_resrgan, 
    progress_resrgan=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Real ESRGAN 🔎]: starting module")
        
    model_resrgan_path  = os.path.join(model_path_resrgan, modelid_resrgan)
    device = torch.device(device_resrgan)
    model_resrgan = RealESRGAN(device, scale=RESRGAN_SCALES[scale_resrgan])
    model_resrgan.load_weights(model_resrgan_path, download=True)
    image = Image.open(img_resrgan).convert('RGB')
    sr_image = model_resrgan.predict(image)
    final_image = [] 
    timestamp = time.time()
    savename = f"outputs/{timestamp}.png"
    
    if use_gfpgan_resrgan == True :
        sr_image = image_gfpgan_mini(sr_image)    
    sr_image.save(savename)
    final_image.append(sr_image)

    print(f">>>[Real ESRGAN 🔎]: generated 1 batch(es) of 1")
    reporting_resrgan = f">>>[Real ESRGAN 🔎]: "+\
        f"Settings : Model={modelid_resrgan} | "+\
        f"Scale={scale_resrgan} | "+\
        f"GFPGAN={use_gfpgan_resrgan} | "
    print(reporting_resrgan)
            
    del model_resrgan, image, sr_image
    clean_ram()

    print(f">>>[Real ESRGAN 🔎]: leaving module")    
    return final_image, final_image
