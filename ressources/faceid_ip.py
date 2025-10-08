# https://github.com/Woolverine94/biniou
# faceid_ip.py
import gradio as gr
import os
import PIL
import cv2
# from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
from photomaker import PhotoMakerStableDiffusionXLPipeline, FaceAnalysis2, analyze_faces
import numpy as np
from huggingface_hub import snapshot_download, hf_hub_download
from compel import Compel, ReturnedEmbeddingsType
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd
import requests

device_label_faceid_ip, model_arch = detect_device()
device_faceid_ip = torch.device(device_label_faceid_ip)

# Gestion des modèles
model_path_faceid_ip = "./models/Stable_Diffusion/"
model_path_ipa_faceid_ip = "./models/Ip-Adapters"
os.makedirs(model_path_faceid_ip, exist_ok=True)
os.makedirs(model_path_ipa_faceid_ip, exist_ok=True)
model_path_community_faceid_ip = "./.community"
os.makedirs(model_path_community_faceid_ip, exist_ok=True)

# if offline_test() == True:
# url_community_faceid_ip = "https://raw.githubusercontent.com/huggingface/diffusers/c0f5346a207bdbf1f7be0b3a539fefae89287ca4/examples/community/ip_adapter_face_id.py"
# response_community_faceid_ip = requests.get(url_community_faceid_ip)
# filename_community_faceid_ip = model_path_community_faceid_ip+ "/ip_adapter_face_id.py"
# with open(filename_community_faceid_ip, "wb") as f:
#     f.write(response_community_faceid_ip.content)

model_list_faceid_ip = []

# .from_single_file NOT compatible with FaceID community pipeline
# for filename in os.listdir(model_path_faceid_ip):
#     f = os.path.join(model_path_faceid_ip, filename)
#     if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
#         model_list_faceid_ip.append(f)

model_list_faceid_ip_builtin = [
#     "SG161222/Realistic_Vision_V3.0_VAE",
#     "SG161222/Paragon_V1.0",
#     "digiplay/majicMIX_realistic_v7",
#     "SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep",
#     "sd-community/sdxl-flash",
#     "dataautogpt3/PrometheusV1",
#     "mann-e/Mann-E_Dreams",
#     "mann-e/Mann-E_Art",
#     "ehristoforu/Visionix-alpha",
#     "RunDiffusion/Juggernaut-X-Hyper",
#     "cutycat2000x/InterDiffusion-4.0",
#     "RunDiffusion/Juggernaut-XL-Lightning",
#     "fluently/Fluently-XL-v3-Lightning",
#     "Corcelio/mobius",
#     "fluently/Fluently-XL-Final",
#     "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep",
#     "recoilme/ColorfulXL-Lightning",
#     "playgroundai/playground-v2-512px-base",
#     "playgroundai/playground-v2-1024px-aesthetic",
#     "playgroundai/playground-v2.5-1024px-aesthetic",
#     "SG161222/RealVisXL_V3.0",
#     "SG161222/RealVisXL_V4.0_Lightning",
#     "cagliostrolab/animagine-xl-3.1",
#     "aipicasso/emi-2",
# #    "stabilityai/sd-turbo",
# #    "stabilityai/sdxl-turbo",
# #    "dataautogpt3/OpenDalleV1.1",
# #    "dataautogpt3/ProteusV0.4",
#     "dataautogpt3/ProteusV0.4-Lightning",
#     "digiplay/AbsoluteReality_v1.8.1",
# #    "segmind/Segmind-Vega",
# #    "segmind/SSD-1B",
#     "gsdf/Counterfeit-V2.5",
# #    "ckpt/anything-v4.5-vae-swapped",
#     "stabilityai/stable-diffusion-xl-base-1.0",
# #    "stabilityai/stable-diffusion-xl-refiner-1.0",
#     "runwayml/stable-diffusion-v1-5",
#     "nitrosocke/Ghibli-Diffusion",

    "-[ 👍 SD15 ]-",
    "SG161222/Realistic_Vision_V3.0_VAE",
    "Yntec/VisionVision",
    "fluently/Fluently-epic",
    "SG161222/Paragon_V1.0",
    "digiplay/AbsoluteReality_v1.8.1",
    "digiplay/majicMIX_realistic_v7",
    "SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep",
    "digiplay/PerfectDeliberate_v5",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "ItsJayQz/GTA5_Artwork_Diffusion",
    "songkey/epicphotogasm_ultimateFidelity",
    "-[ 👍 🇯🇵 Anime SD15 ]-",
    "gsdf/Counterfeit-V2.5",
    "fluently/Fluently-anime",
    "xyn-ai/anything-v4.0",
    "nitrosocke/Ghibli-Diffusion",
    "digiplay/STRANGER-ANIME",
    "Norod78/sd15-jojo-stone-ocean",
    "stablediffusionapi/anything-v5",
    "-[ 👌 🐢 SDXL ]-",
    "fluently/Fluently-XL-Final",
    "SG161222/RealVisXL_V5.0",
    "Corcelio/mobius",
    "misri/juggernautXL_juggXIByRundiffusion",
    "mann-e/Mann-E_Dreams",
    "mann-e/Mann-E_Art",
    "ehristoforu/Visionix-alpha",
    "cutycat2000x/InterDiffusion-4.0",
    "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep",
    "GraydientPlatformAPI/flashback-xl",
    "dataautogpt3/Proteus-v0.6",
    "dataautogpt3/PrometheusV1",
    "dataautogpt3/ProteusSigma",
    "Chan-Y/Stable-Flash-Lightning",
    "stablediffusionapi/protovision-xl-high-fidel",
    "comin/IterComp",
    "Spestly/OdysseyXL-1.0",
    "eramth/realism-sdxl",
    "yandex/stable-diffusion-xl-base-1.0-alchemist",
    "John6666/stellaratormix-photorealism-v30-sdxl",
    "RunDiffusion/Juggernaut-XL-v6",
    "playgroundai/playground-v2-512px-base",
    "playgroundai/playground-v2-1024px-aesthetic",
    "playgroundai/playground-v2.5-1024px-aesthetic",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "-[ 👌 🚀 Fast SDXL ]-",
    "sd-community/sdxl-flash",
    "fluently/Fluently-XL-v3-Lightning",
    "GraydientPlatformAPI/epicrealism-lightning-xl",
    "Lykon/dreamshaper-xl-lightning",
    "RunDiffusion/Juggernaut-XL-Lightning",
    "RunDiffusion/Juggernaut-X-Hyper",
    "SG161222/RealVisXL_V5.0_Lightning",
    "dataautogpt3/ProteusV0.4-Lightning",
    "recoilme/ColorfulXL-Lightning",
    "GraydientPlatformAPI/lustify-lightning",
    "John6666/comradeship-xl-v9a-spo-dpo-flash-sdxl",
    "stablediffusionapi/dream-diffusion-lightning",
    "John6666/jib-mix-realistic-xl-v15-maximus-sdxl",
    "muverqqw/Dreamcoil-lightning",
    "-[ 👌 🇯🇵 Anime SDXL ]-",
    "GraydientPlatformAPI/geekpower-cellshade-xl",
    "cagliostrolab/animagine-xl-4.0",
    "Bakanayatsu/ponyDiffusion-V6-XL-Turbo-DPO",
    "OnomaAIResearch/Illustrious-xl-early-release-v0",
    "GraydientPlatformAPI/sanae-xl",
    "yodayo-ai/clandestine-xl-1.0",
    "stablediffusionapi/anime-journey-v2",
    "aipicasso/emi-2",
    "zenless-lab/sdxl-anything-xl",
]

for k in range(len(model_list_faceid_ip_builtin)):
    model_list_faceid_ip.append(model_list_faceid_ip_builtin[k])

# Bouton Cancel
stop_faceid_ip = False

def initiate_stop_faceid_ip() :
    global stop_faceid_ip
    stop_faceid_ip = True

def check_faceid_ip(pipe, step_index, timestep, callback_kwargs): 
    global stop_faceid_ip
    if stop_faceid_ip == True:
        print(">>>[Photobooth 🖼️ ]: generation canceled by user")
        stop_faceid_ip = False
        pipe._interrupt = True
    return callback_kwargs

# def face_extractor(image_src):
#     app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# #    app.prepare(ctx_id=0, det_size=(640, 640))
#     app.prepare(ctx_id=0, det_size=(320, 320))
#     
#     image = cv2.imread(image_src)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     faces = app.get(image)
#     faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
#     return faceid_embeds

def face_analyser(input_id_images):
    face_detector = FaceAnalysis2(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
    face_detector.prepare(ctx_id=0, det_size=(640, 640))
    id_embed_list = []
    for img in input_id_images:
        img = np.array(img)
        img = img[:, :, ::-1]
        faces = analyze_faces(face_detector, img)
        if len(faces) > 0:
            id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))
    if len(id_embed_list) == 0:
        raise ValueError(f"No face detected in input image pool")
    id_embeds = torch.stack(id_embed_list)
    return id_embeds

@metrics_decoration
def image_faceid_ip(
    modelid_faceid_ip, 
    sampler_faceid_ip, 
    img_faceid_ip, 
    prompt_faceid_ip, 
    negative_prompt_faceid_ip, 
    num_images_per_prompt_faceid_ip, 
    num_prompt_faceid_ip, 
    guidance_scale_faceid_ip, 
    denoising_strength_faceid_ip, 
    num_inference_step_faceid_ip, 
    height_faceid_ip, 
    width_faceid_ip, 
    seed_faceid_ip, 
    use_gfpgan_faceid_ip, 
    nsfw_filter, 
    tkme_faceid_ip,
    clipskip_faceid_ip,
    lora_model_faceid_ip,
    lora_weight_faceid_ip,
    lora_model2_faceid_ip,
    lora_weight2_faceid_ip,
    lora_model3_faceid_ip,
    lora_weight3_faceid_ip,
    lora_model4_faceid_ip,
    lora_weight4_faceid_ip,
    lora_model5_faceid_ip,
    lora_weight5_faceid_ip,
    txtinv_faceid_ip,
    progress_faceid_ip=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Photobooth 🖼️ ]: starting module")
    print(f">>>[Photobooth 🖼️ ]: generated {num_prompt_faceid_ip} batch(es) of {num_images_per_prompt_faceid_ip}")
    print(f">>>[Photobooth 🖼️ ]: leaving module")
    return ["dummy.png"], ["dummy.png"]
