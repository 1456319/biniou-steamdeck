# https://github.com/Woolverine94/biniou
# img2img.py
import gradio as gr
import os
import PIL
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline, AutoPipelineForImage2Image, StableDiffusion3Img2ImgPipeline, FluxImg2ImgPipeline
from huggingface_hub import hf_hub_download
from compel import Compel, ReturnedEmbeddingsType
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd
from diffusers.schedulers import AysSchedules

device_label_img2img, model_arch = detect_device()
device_img2img = torch.device(device_label_img2img)

# Gestion des modèles
model_path_img2img = "./models/Stable_Diffusion/"
os.makedirs(model_path_img2img, exist_ok=True)
model_path_flux_img2img = "./models/Flux/"
os.makedirs(model_path_flux_img2img, exist_ok=True)

model_list_img2img_local = []

for filename in os.listdir(model_path_img2img):
    f = os.path.join(model_path_img2img, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_img2img_local.append(f)

model_list_img2img_builtin = [
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
    "-[ 👍 🚀 Fast SD15 ]-",
    "IDKiro/sdxs-512-0.9",
    "IDKiro/sdxs-512-dreamshaper",
    "stabilityai/sd-turbo",
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
    "etri-vilab/koala-lightning-700m",
    "etri-vilab/koala-lightning-1b",
    "GraydientPlatformAPI/flashback-xl",
    "dataautogpt3/ProteusV0.5",
    "dataautogpt3/Proteus-v0.6",
    "dataautogpt3/PrometheusV1",
    "dataautogpt3/OpenDalleV1.1",
    "dataautogpt3/ProteusSigma",
    "Chan-Y/Stable-Flash-Lightning",
    "stablediffusionapi/protovision-xl-high-fidel",
    "comin/IterComp",
    "Spestly/OdysseyXL-1.0",
    "eramth/realism-sdxl",
    "yandex/stable-diffusion-xl-base-1.0-alchemist",
    "John6666/stellaratormix-photorealism-v30-sdxl",
    "RunDiffusion/Juggernaut-XL-v6",
    "segmind/SSD-1B",
    "segmind/Segmind-Vega",
    "playgroundai/playground-v2-512px-base",
    "playgroundai/playground-v2-1024px-aesthetic",
    "playgroundai/playground-v2.5-1024px-aesthetic",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
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
    "thibaud/sdxl_dpo_turbo",
    "stabilityai/sdxl-turbo",
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
    "-[ 👏 🐢 SD3 ]-",
    "v2ray/stable-diffusion-3-medium-diffusers",
    "ptx0/sd3-reality-mix",
    "-[ 👏 🐢 SD3.5 Large ]-",
    "adamo1139/stable-diffusion-3.5-large-turbo-ungated",
    "ariG23498/sd-3.5-merged",
    "aipicasso/emi-3",
    "-[ 👏 🐢 SD3.5 Medium ]-",
    "adamo1139/stable-diffusion-3.5-medium-ungated",
    "tensorart/stable-diffusion-3.5-medium-turbo",
    "-[ 🏆 🐢 Flux ]-",
    "Freepik/flux.1-lite-8B",
    "black-forest-labs/FLUX.1-schnell",
    "sayakpaul/FLUX.1-merged",
    "ChuckMcSneed/FLUX.1-dev",
    "NikolaSigmoid/FLUX.1-Krea-dev",
    "AlekseyCalvin/FluxKrea_HSTurbo_Diffusers",
    "minpeter/FLUX-Hyperscale-fused",
    "enhanceaiteam/Mystic",
    "AlekseyCalvin/AuraFlux_merge_diffusers",
    "ostris/Flex.1-alpha",
    "shuttleai/shuttle-jaguar",
    "Shakker-Labs/AWPortrait-FL",
    "AlekseyCalvin/PixelWave_Schnell_03_by_humblemikey_Diffusers_fp8_T4bf16",
    "AlekseyCalvin/PixelwaveFluxSchnell_Diffusers",
    "mikeyandfriends/PixelWave_FLUX.1-schnell_04",
    "minpeter/FLUX-Hyperscale-fused-fast",
    "-[ 🏠 Local models ]-",
]

model_list_img2img = model_list_img2img_builtin

for k in range(len(model_list_img2img_local)):
    model_list_img2img.append(model_list_img2img_local[k])

# Bouton Cancel
stop_img2img = False

def initiate_stop_img2img() :
    global stop_img2img
    stop_img2img = True

def check_img2img(pipe, step_index, timestep, callback_kwargs) : 
    global stop_img2img
    if stop_img2img == True :
        print(">>>[img2img 🖌️ ]: generation canceled by user")
        stop_img2img = False
        pipe._interrupt = True
    return callback_kwargs

@metrics_decoration
def image_img2img(
    modelid_img2img, 
    sampler_img2img, 
    img_img2img, 
    prompt_img2img, 
    negative_prompt_img2img, 
    num_images_per_prompt_img2img, 
    num_prompt_img2img, 
    guidance_scale_img2img, 
    denoising_strength_img2img, 
    num_inference_step_img2img, 
    height_img2img, 
    width_img2img, 
    seed_img2img, 
    source_type_img2img, 
    use_gfpgan_img2img, 
    nsfw_filter, 
    tkme_img2img,
    clipskip_img2img,
    use_ays_img2img,
    lora_model_img2img,
    lora_weight_img2img,
    lora_model2_img2img,
    lora_weight2_img2img,
    lora_model3_img2img,
    lora_weight3_img2img,
    lora_model4_img2img,
    lora_weight4_img2img,
    lora_model5_img2img,
    lora_weight5_img2img,
    txtinv_img2img,
    progress_img2img=gr.Progress(track_tqdm=True)
    ):

    print(">>>[img2img 🖌️ ]: starting module")
    print(f">>>[img2img 🖌️ ]: generated {num_prompt_img2img} batch(es) of {num_images_per_prompt_img2img}")
    print(f">>>[img2img 🖌️ ]: leaving module")
    return ["dummy.png"], ["dummy.png"]
