# https://github.com/Woolverine94/biniou
# controlnet.py
import gradio as gr
import os
import cv2
import torch
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, StableDiffusion3ControlNetPipeline, FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from huggingface_hub import hf_hub_download
from compel import Compel, ReturnedEmbeddingsType
import random
from ressources.gfpgan import *
from controlnet_aux.processor import Processor
import tomesd
from diffusers.schedulers import AysSchedules

device_label_controlnet, model_arch = detect_device()
device_controlnet = torch.device(device_label_controlnet)

# Gestion des modèles
model_path_controlnet = "./models/Stable_Diffusion/"
model_path_flux_controlnet = "./models/Flux/"
os.makedirs(model_path_controlnet, exist_ok=True)
os.makedirs(model_path_flux_controlnet, exist_ok=True)
model_list_controlnet_local = []

for filename in os.listdir(model_path_controlnet):
    f = os.path.join(model_path_controlnet, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_controlnet_local.append(f)

model_list_controlnet_builtin = [
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
    "stabilityai/stable-diffusion-xl-base-1.0",
    "-[ 👌 🚀 Fast SDXL ]-",
    "sd-community/sdxl-flash",
    "fluently/Fluently-XL-v3-Lightning",
    "GraydientPlatformAPI/epicrealism-lightning-xl",
    "Lykon/dreamshaper-xl-lightning",
    "RunDiffusion/Juggernaut-XL-Lightning",
    "RunDiffusion/Juggernaut-X-Hyper",
    "SG161222/RealVisXL_V5.0_Lightning",
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

model_list_controlnet = model_list_controlnet_builtin

for k in range(len(model_list_controlnet_local)):
    model_list_controlnet.append(model_list_controlnet_local[k])

model_path_base_controlnet = "./models/controlnet"
os.makedirs(model_path_base_controlnet, exist_ok=True)

base_controlnet = "lllyasviel/ControlNet-v1-1"

variant_list_controlnet = [
    "lllyasviel/control_v11p_sd15_canny",
    "lllyasviel/control_v11f1p_sd15_depth",
    "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "lllyasviel/control_v11p_sd15_lineart",
    "lllyasviel/control_v11p_sd15_mlsd",
    "lllyasviel/control_v11p_sd15_normalbae",
    "lllyasviel/control_v11p_sd15_openpose",
    "lllyasviel/control_v11p_sd15_scribble",
    "lllyasviel/control_v11p_sd15_softedge",
    "lllyasviel/control_v11f1e_sd15_tile",
    "Nacholmo/controlnet-qr-pattern-v2",
    "monster-labs/control_v1p_sd15_qrcode_monster",
    "patrickvonplaten/controlnet-canny-sdxl-1.0",
    "patrickvonplaten/controlnet-depth-sdxl-1.0",
    "thibaud/controlnet-openpose-sdxl-1.0",
    "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
    "ValouF-pimento/ControlNet_SDXL_tile_upscale",
    "Nacholmo/controlnet-qr-pattern-sdxl",
    "monster-labs/control_v1p_sdxl_qrcode_monster",
    "TheMistoAI/MistoLine",
    "brad-twinkl/controlnet-union-sdxl-1.0-promax",
    "xinsir/controlnet-union-sdxl-1.0",
    "InstantX/SD3-Controlnet-Canny",
    "InstantX/SD3-Controlnet-Pose",
    "InstantX/SD3-Controlnet-Tile",
    "XLabs-AI/flux-controlnet-canny-diffusers",
    "XLabs-AI/flux-controlnet-depth-diffusers",
#    "George0667/Flux.1-dev-ControlNet-LineCombo",
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
    "InstantX/FLUX.1-dev-Controlnet-Union",
#    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
]

preprocessor_list_controlnet = [
    "canny",
    "depth_leres",
    "depth_leres++",
    "depth_midas",
    "lineart_anime",
    "lineart_coarse",
    "lineart_realistic",
    "mlsd",
    "normal_bae",
    "openpose",
    "openpose_face",
    "openpose_faceonly",
    "openpose_full",
    "openpose_hand",
    "scribble_hed",
    "scribble_pidinet",
    "softedge_hed",
    "softedge_hedsafe",
    "softedge_pidinet",
    "softedge_pidsafe",
    "tile",
    "qr",
    "qr_invert",
    "qr_monster",
    "qr_monster_invert",
]

# Bouton Cancel
stop_controlnet = False

def initiate_stop_controlnet() :
    global stop_controlnet
    stop_controlnet = True

def check_controlnet(pipe, step_index, timestep, callback_kwargs) :
    global stop_controlnet
    if stop_controlnet == False :
        return callback_kwargs
    elif stop_controlnet == True :
        print(">>>[ControlNet 🖼️ ]: generation canceled by user")
        stop_controlnet = False
        try:
            del ressources.controlnet.pipe_controlnet
        except NameError as e:
            raise Exception("Interrupting ...")
            return "Canceled ..."
    return

def dispatch_controlnet_preview(
    modelid_controlnet,
    low_threshold_controlnet,
    high_threshold_controlnet,
    img_source_controlnet,
    preprocessor_controlnet,
    progress_controlnet=gr.Progress(track_tqdm=True)
    ):

    modelid_controlnet = model_cleaner_sd(modelid_controlnet)

    if is_sdxl(modelid_controlnet):
        is_xl_controlnet: bool = True
    else :
        is_xl_controlnet: bool = False

    if is_sd3(modelid_controlnet):
        is_sd3_controlnet: bool = True
    else :
        is_sd3_controlnet: bool = False

    if is_flux(modelid_controlnet):
        is_flux_controlnet: bool = True
    else :
        is_flux_controlnet: bool = False

    img_source_controlnet = Image.open(img_source_controlnet)
    img_source_controlnet = np.array(img_source_controlnet)
    if not (('qr' in preprocessor_controlnet) or ('tile' in preprocessor_controlnet)):
        processor_controlnet = Processor(preprocessor_controlnet)

#    match preprocessor_controlnet:
# 01
    if preprocessor_controlnet == "canny":
        result = canny_controlnet(img_source_controlnet, low_threshold_controlnet, high_threshold_controlnet)
#        return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[0]
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20]
        elif is_sd3_controlnet:
            return result, result, variant_list_controlnet[22]
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25]
        else:
            return result, result, variant_list_controlnet[0]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[0]
# 02
    elif preprocessor_controlnet == "depth_leres":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[13] 
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[26] 
        else:
            return result, result, variant_list_controlnet[1]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[1]
# 03
    elif preprocessor_controlnet == "depth_leres++":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[13] 
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[26] 
        else:
            return result, result, variant_list_controlnet[1]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[1]
# 04
    elif preprocessor_controlnet == "depth_midas":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
#        return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[1]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[1]
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20] 
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[26] 
        else:
            return result, result, variant_list_controlnet[1]
#     case "depth_zoe":
#         result = processor_controlnet(img_source_controlnet, to_pil=True)
# #        return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[1]
#         return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[1]
# 05
    elif preprocessor_controlnet == "lineart_anime":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
#        return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[2]
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20]
        elif is_sd3_controlnet:
            return result, result, variant_list_controlnet[22]
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25]
        else:
            return result, result, variant_list_controlnet[2]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[2]
# 06
    elif preprocessor_controlnet == "lineart_coarse":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
#        return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[3]
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[12]
        elif is_sd3_controlnet:
            return result, result, variant_list_controlnet[22]
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25]
        else:
            return result, result, variant_list_controlnet[3]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[3]
# 07
    elif preprocessor_controlnet == "lineart_realistic":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
#        return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[3]
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20]
        elif is_sd3_controlnet:
            return result, result, variant_list_controlnet[22]
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25]
        else:
            return result, result, variant_list_controlnet[3]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[3]
# 08
    elif preprocessor_controlnet == "mlsd":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
#        return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[4]
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20]
        elif is_sd3_controlnet:
            return result, result, variant_list_controlnet[22]
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25]
        else:
            return result, result, variant_list_controlnet[4]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[4]
# 09
    elif preprocessor_controlnet == "normal_bae":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
#        return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[5]
        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[5]
#     case "normal_midas":
#         result = processor_controlnet(img_source_controlnet, to_pil=True)
# #        return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[5]
#         return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[5]
# 10
    elif preprocessor_controlnet == "openpose":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
#        return result, result, variant_list_controlnet[14] if is_xl_controlnet else variant_list_controlnet[6]
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20]
        elif is_sd3_controlnet:
            return result, result, variant_list_controlnet[23]
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25]
        else:
            return result, result, variant_list_controlnet[6]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[6]
# 11
    elif preprocessor_controlnet == "openpose_face":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
#        return result, result, variant_list_controlnet[14] if is_xl_controlnet else variant_list_controlnet[6]
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[14]
        elif is_sd3_controlnet:
            return result, result, variant_list_controlnet[23]
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25]
        else:
            return result, result, variant_list_controlnet[6]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[6]
# 12
    elif preprocessor_controlnet == "openpose_faceonly":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
#        return result, result, variant_list_controlnet[14] if is_xl_controlnet else variant_list_controlnet[6]
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[14]
        elif is_sd3_controlnet:
            return result, result, variant_list_controlnet[23]
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25]
        else:
            return result, result, variant_list_controlnet[6]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[6]
# 13
    elif preprocessor_controlnet == "openpose_full":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
#        return result, result, variant_list_controlnet[14] if is_xl_controlnet else variant_list_controlnet[6]
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[14]
        elif is_sd3_controlnet:
            return result, result, variant_list_controlnet[23]
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25]
        else:
            return result, result, variant_list_controlnet[6]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[6]
# 14
    elif preprocessor_controlnet == "openpose_hand":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
#        return result, result, variant_list_controlnet[14] if is_xl_controlnet else variant_list_controlnet[6]
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20]
        elif is_sd3_controlnet:
            return result, result, variant_list_controlnet[23]
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25]
        else:
            return result, result, variant_list_controlnet[6]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[6]
	
# 15
    elif preprocessor_controlnet == "scribble_hed":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20] 
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25] 
        else:
            return result, result, variant_list_controlnet[7]
#        return result, result, variant_list_controlnet[15] if is_xl_controlnet else variant_list_controlnet[7]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[7]
# 16
    elif preprocessor_controlnet == "scribble_pidinet":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20] 
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25] 
        else:
            return result, result, variant_list_controlnet[7]
#        return result, result, variant_list_controlnet[15] if is_xl_controlnet else variant_list_controlnet[7]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[7]
# 17
    elif preprocessor_controlnet == "softedge_hed":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20]
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25]
        else:
            return result, result, variant_list_controlnet[8]
#        return result, result, variant_list_controlnet[15] if is_xl_controlnet else variant_list_controlnet[8]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[8]
# 18
    elif preprocessor_controlnet == "softedge_hedsafe":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20] 
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25] 
        else:
            return result, result, variant_list_controlnet[8]
#        return result, result, variant_list_controlnet[15] if is_xl_controlnet else variant_list_controlnet[8]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[8]
# 19
    elif preprocessor_controlnet == "softedge_pidinet":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20] 
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25] 
        else:
            return result, result, variant_list_controlnet[8]
#        return result, result, variant_list_controlnet[15] if is_xl_controlnet else variant_list_controlnet[8]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[8]
# 20
    elif preprocessor_controlnet == "softedge_pidsafe":
        result = processor_controlnet(img_source_controlnet, to_pil=True)
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20] 
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[25] 
        else:
            return result, result, variant_list_controlnet[8]
#        return result, result, variant_list_controlnet[15] if is_xl_controlnet else variant_list_controlnet[8]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[8]
# 21
    elif preprocessor_controlnet == "tile":
        result = tile_controlnet(img_source_controlnet, is_xl_controlnet, modelid_controlnet)
#        return result, result, variant_list_controlnet[16] if is_xl_controlnet else variant_list_controlnet[9]
        if is_xl_controlnet:
            return result, result, variant_list_controlnet[20]
        elif is_sd3_controlnet:
            return result, result, variant_list_controlnet[24]
        elif is_flux_controlnet:
            return result, result, variant_list_controlnet[27]
        else:
            return result, result, variant_list_controlnet[9]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[9]
# 22
    elif preprocessor_controlnet == "qr":
        result = qr_controlnet(img_source_controlnet, 0)
        return result, result, variant_list_controlnet[17] if is_xl_controlnet else variant_list_controlnet[10]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[10]
# 23
    elif preprocessor_controlnet == "qr_invert":
        result = qr_controlnet(img_source_controlnet, 1)
        return result, result, variant_list_controlnet[17] if is_xl_controlnet else variant_list_controlnet[10]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[10]
# 24
    elif preprocessor_controlnet == "qr_monster":
        result = qr_controlnet(img_source_controlnet, 0)
        return result, result, variant_list_controlnet[18] if is_xl_controlnet else variant_list_controlnet[11]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[11]
# 25
    elif preprocessor_controlnet == "qr_monster_invert":
        result = qr_controlnet(img_source_controlnet, 1)
        return result, result, variant_list_controlnet[18] if is_xl_controlnet else variant_list_controlnet[11]
#        return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[11]

def canny_controlnet(image, low_threshold, high_threshold):
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def tile_controlnet(image, is_xl_controlnet, modelid_controlnet):
    image = Image.fromarray(image)
    width, height = image.size
    if (is_xl_controlnet) and ("TURBO" not in modelid_controlnet.upper()):
        dim_size = correct_size(width, height, 1024)
    else :
        dim_size = correct_size(width, height, 512)
    image = image.convert("RGB")
    image = image.resize((dim_size[0], dim_size[1]), resample=Image.LANCZOS)
    return image

def qr_controlnet(image, switch):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    if (switch == 1):
        image = cv2.bitwise_not(image)
    return image

@metrics_decoration
def image_controlnet(
    modelid_controlnet,
    sampler_controlnet,
    prompt_controlnet,
    negative_prompt_controlnet,
    num_images_per_prompt_controlnet,
    num_prompt_controlnet,
    guidance_scale_controlnet,
    num_inference_step_controlnet,
    height_controlnet,
    width_controlnet,
    seed_controlnet,
    low_threshold_controlnet,
    high_threshold_controlnet,
    strength_controlnet,
    start_controlnet,
    stop_controlnet,
    use_gfpgan_controlnet,
    preprocessor_controlnet,
    variant_controlnet,
    img_preview_controlnet,
    nsfw_filter, 
    tkme_controlnet,
    clipskip_controlnet,
    use_ays_controlnet,
    lora_model_controlnet,
    lora_weight_controlnet,
    lora_model2_controlnet,
    lora_weight2_controlnet,
    lora_model3_controlnet,
    lora_weight3_controlnet,
    lora_model4_controlnet,
    lora_weight4_controlnet,
    lora_model5_controlnet,
    lora_weight5_controlnet,
    txtinv_controlnet,
    progress_controlnet=gr.Progress(track_tqdm=True)
    ):

    print(">>>[ControlNet 🖼️ ]: starting module")
    print(f">>>[ControlNet 🖼️ ]: generated {num_prompt_controlnet} batch(es) of {num_images_per_prompt_controlnet}")
    print(f">>>[ControlNet 🖼️ ]: leaving module")
    return ["dummy.png"], ["dummy.png"]
