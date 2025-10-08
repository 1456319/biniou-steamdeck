# https://github.com/Woolverine94/biniou
# faceswapper.py
import os
import cv2
import copy
import insightface
import onnxruntime
import numpy as np
from PIL import Image
import random
from ressources.common import *
from ressources.gfpgan import *
from huggingface_hub import snapshot_download, hf_hub_download

# Gestion des modèles
model_path_faceswap = "./models/faceswap/"
os.makedirs(model_path_faceswap, exist_ok=True)

model_list_faceswap = {}

# for filename in os.listdir(model_path_faceswap):
#     f = os.path.join(model_path_faceswap, filename)
#     if os.path.isfile(f) and filename.endswith('.onnx') :
#         print(filename, f)
#         model_list_faceswap.update({f: ""})

model_list_faceswap_builtin = {
    "thebiglaskowski/inswapper_128.onnx": "inswapper_128.onnx",
}

model_list_faceswap.update(model_list_faceswap_builtin)

def download_model(modelid_faceswap):
    if modelid_faceswap[0:9] != "./models/":
        hf_hub_path_faceswap = hf_hub_download(
            repo_id=modelid_faceswap, 
            filename=model_list_faceswap[modelid_faceswap], 
            repo_type="model", 
            cache_dir=model_path_faceswap, 
            local_dir=model_path_faceswap, 
            local_dir_use_symlinks=True, 
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        modelid_faceswap = hf_hub_path_faceswap
    return modelid_faceswap    

@metrics_decoration
def image_faceswap(
    modelid_faceswap, 
    img_source_faceswap, 
    img_target_faceswap, 
    source_index_faceswap, 
    target_index_faceswap, 
    use_gfpgan_faceswap, 
    progress_faceswap=gr.Progress(track_tqdm=True)
    ):
   
    print(">>>[Faceswap 🎭 ]: starting module")
    print(f">>>[Faceswap 🎭 ]: generated 1 batch(es) of 1")
    print(f">>>[Faceswap 🎭 ]: leaving module")
    return ["dummy.png"], ["dummy.png"]
