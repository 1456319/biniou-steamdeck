# https://github.com/Woolverine94/biniou
# faceid_ip.py
import gradio as gr
import os
import PIL
import cv2
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
from huggingface_hub import snapshot_download, hf_hub_download
from compel import Compel, ReturnedEmbeddingsType
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_label_faceid_ip, model_arch = detect_device()
device_faceid_ip = torch.device(device_label_faceid_ip)

# Gestion des modèles
model_path_faceid_ip = "./models/Stable_Diffusion/"
model_path_ipa_faceid_ip = "./models/Ip-Adapters"
os.makedirs(model_path_faceid_ip, exist_ok=True)
os.makedirs(model_path_ipa_faceid_ip, exist_ok=True)

model_list_faceid_ip = []

for filename in os.listdir(model_path_faceid_ip):
    f = os.path.join(model_path_faceid_ip, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_faceid_ip.append(f)

model_list_faceid_ip_builtin = [
    "SG161222/Realistic_Vision_V3.0_VAE",
#    "stabilityai/sd-turbo",
#    "stabilityai/sdxl-turbo",
#    "dataautogpt3/OpenDalleV1.1",
    "digiplay/AbsoluteReality_v1.8.1",
#    "segmind/Segmind-Vega",
#    "segmind/SSD-1B",
#    "ckpt/anything-v4.5-vae-swapped",
#    "stabilityai/stable-diffusion-xl-base-1.0",
#    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "runwayml/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion",
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
        print(">>>[IP-Adapter 🖌️ ]: generation canceled by user")
        stop_faceid_ip = False
        pipe._interrupt = True
    return callback_kwargs

def face_extractor(image_src):
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#    app.prepare(ctx_id=0, det_size=(640, 640))
    app.prepare(ctx_id=0, det_size=(320, 320))
    
    image = cv2.imread(image_src)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = app.get(image)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    return faceid_embeds

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
    lora_model_faceid_ip,
    lora_weight_faceid_ip,
    progress_faceid_ip=gr.Progress(track_tqdm=True)
    ):

    print(">>>[IP-Adapter 🖌️ ]: starting module")

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_faceid_ip, device_faceid_ip, nsfw_filter)

    if (modelid_faceid_ip == "stabilityai/sdxl-turbo") or (modelid_faceid_ip == "stabilityai/sd-turbo"):
        is_xlturbo_faceid_ip: bool = True
    else :
        is_xlturbo_faceid_ip: bool = False

    if (('xl' or 'XL' or 'Xl' or 'xL') in modelid_faceid_ip or (modelid_faceid_ip == "segmind/SSD-1B") or (modelid_faceid_ip == "segmind/Segmind-Vega") or (modelid_faceid_ip == "dataautogpt3/OpenDalleV1.1")):
        is_xl_faceid_ip: bool = True
    else :
        is_xl_faceid_ip: bool = False     

    ip_ckpt_faceid_ip = f"{model_path_ipa_faceid_ip}/ip-adapter-faceid_sdxl.bin" if is_xl_faceid_ip else f"{model_path_ipa_faceid_ip}/ip-adapter-faceid_sd15.bin"

#    if which_os() == "win32":
    if (is_xl_faceid_ip == True):
        hf_hub_download(
            repo_id="h94/IP-Adapter-FaceID", 
            filename="ip-adapter-faceid_sdxl.bin",
            repo_type="model",
            local_dir=model_path_ipa_faceid_ip,
            local_dir_use_symlinks=False,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
    else:
        hf_hub_download(
            repo_id="h94/IP-Adapter-FaceID", 
            filename="ip-adapter-faceid_sd15.bin",
            repo_type="model",
            local_dir=model_path_ipa_faceid_ip,
            local_dir_use_symlinks=False,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )

    if (is_xlturbo_faceid_ip == True) :
        if modelid_faceid_ip[0:9] == "./models/" :
            pipe_faceid_ip = AutoPipelineForText2Image.from_single_file(
                modelid_faceid_ip, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                custom_pipeline="ip_adapter_face_id",
            )
        else :        
            pipe_faceid_ip = AutoPipelineForText2Image.from_pretrained(
                modelid_faceid_ip, 
                cache_dir=model_path_faceid_ip, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                custom_pipeline="ip_adapter_face_id",
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    elif (is_xl_faceid_ip == True) and (is_xlturbo_faceid_ip == False) :
        if modelid_faceid_ip[0:9] == "./models/" :
            pipe_faceid_ip = StableDiffusionXLPipeline.from_single_file(
                modelid_faceid_ip, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                custom_pipeline="ip_adapter_face_id",
            )
        else :        
            pipe_faceid_ip = StableDiffusionXLPipeline.from_pretrained(
                modelid_faceid_ip, 
                cache_dir=model_path_faceid_ip, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                custom_pipeline="ip_adapter_face_id",
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    else :
        if modelid_faceid_ip[0:9] == "./models/" :
            pipe_faceid_ip = StableDiffusionPipeline.from_single_file(
                modelid_faceid_ip, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                custom_pipeline="ip_adapter_face_id",
            )
        else :        
            pipe_faceid_ip = StableDiffusionPipeline.from_pretrained(
                modelid_faceid_ip, 
                cache_dir=model_path_faceid_ip, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                custom_pipeline="ip_adapter_face_id",
                resume_download=True,
                local_files_only=True if offline_test() else None
            )

#    if (is_xl_faceid_ip == True) or (is_xlturbo_faceid_ip == True):

#    if (which_os() == "win32"):
#        if (is_xl_faceid_ip == True):
#            pipe_faceid_ip.load_ip_adapter(
#                model_path_ipa_faceid_ip,
#                subfolder="",
#                weight_name="ip-adapter-faceid_sdxl.bin",
#                torch_dtype=model_arch,
#                use_safetensors=True,
#                resume_download=True,
#                local_files_only=True if offline_test() else None
#            )
#        else:
#            pipe_faceid_ip.load_ip_adapter(
#                model_path_ipa_faceid_ip,
#                subfolder="",
#                weight_name="ip-adapter-faceid_sd15.bin",
#                torch_dtype=model_arch,
#                use_safetensors=True, 
#                resume_download=True,
#                local_files_only=True if offline_test() else None
#            )
#    else:
#        if (is_xl_faceid_ip == True):
#            pipe_faceid_ip.load_ip_adapter(
#                "h94/IP-Adapter-FaceID", 
#                cache_dir=model_path_ipa_faceid_ip,
#                subfolder="",
#                weight_name="ip-adapter-faceid_sdxl.bin",
#                torch_dtype=model_arch,
#                use_safetensors=True,
#                resume_download=True,
#                local_files_only=True if offline_test() else None
#            )
#        else:
#            pipe_faceid_ip.load_ip_adapter(
#                "h94/IP-Adapter-FaceID",
#                cache_dir=model_path_ipa_faceid_ip,
#                subfolder="",
#                weight_name="ip-adapter-faceid_sd15.bin",
#                torch_dtype=model_arch,
#                use_safetensors=True, 
#                resume_download=True,
#                local_files_only=True if offline_test() else None
#            )

        if (is_xl_faceid_ip == True):
            pipe_faceid_ip.load_ip_adapter_face_id("h94/IP-Adapter-FaceID", weight_name="ip-adapter-faceid_sdxl.bin")
        else:
            pipe_faceid_ip.load_ip_adapter_face_id("h94/IP-Adapter-FaceID", weight_name="ip-adapter-faceid_sd15.bin")

    pipe_faceid_ip.set_ip_adapter_scale(denoising_strength_faceid_ip)
    pipe_faceid_ip = schedulerer(pipe_faceid_ip, sampler_faceid_ip)
#    pipe_faceid_ip.enable_attention_slicing("max")  
    tomesd.apply_patch(pipe_faceid_ip, ratio=tkme_faceid_ip)
    if device_label_faceid_ip == "cuda" :
        pipe_faceid_ip.enable_sequential_cpu_offload()
    else : 
        pipe_faceid_ip = pipe_faceid_ip.to(device_faceid_ip)

    if lora_model_faceid_ip != "":
        model_list_lora_faceid_ip = lora_model_list(modelid_faceid_ip)
        if modelid_faceid_ip[0:9] == "./models/":
            pipe_faceid_ip.load_lora_weights(
                os.path.dirname(lora_model_faceid_ip),
                weight_name=model_list_lora_faceid_ip[lora_model_faceid_ip][0],
                use_safetensors=True,
                adapter_name="adapter1",
            )
        else:
            if is_xl_faceid_ip:
                lora_model_path = "./models/lora/SDXL"
            else: 
                lora_model_path = "./models/lora/SD"
            pipe_faceid_ip.load_lora_weights(
                lora_model_faceid_ip,
                weight_name=model_list_lora_faceid_ip[lora_model_faceid_ip][0],
                cache_dir=lora_model_path,
                use_safetensors=True,
                adapter_name="adapter1",
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        pipe_faceid_ip.fuse_lora(lora_scale=lora_weight_faceid_ip)
#        pipe_faceid_ip.set_adapters(["adapter1"], adapter_weights=[float(lora_weight_faceid_ip)])

#    ip_model_faceid_ip = IPAdapterFaceID(pipe_faceid_ip, ip_ckpt_faceid_ip, device_faceid_ip)

    if seed_faceid_ip == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_faceid_ip)

    prompt_faceid_ip = str(prompt_faceid_ip)
    negative_prompt_faceid_ip = str(negative_prompt_faceid_ip)
    if prompt_faceid_ip == "None":
        prompt_faceid_ip = ""
    if negative_prompt_faceid_ip == "None":
        negative_prompt_faceid_ip = ""

    if (is_xl_faceid_ip == True) :
        compel = Compel(
            tokenizer=pipe_faceid_ip.tokenizer_2, 
            text_encoder=pipe_faceid_ip.text_encoder_2, 
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
            requires_pooled=[False, True], 
        )
        conditioning, pooled = compel(prompt_faceid_ip)
        neg_conditioning, neg_pooled = compel(negative_prompt_faceid_ip)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    else :
        compel = Compel(tokenizer=pipe_faceid_ip.tokenizer, text_encoder=pipe_faceid_ip.text_encoder, truncate_long_prompts=False)
        conditioning = compel.build_conditioning_tensor(prompt_faceid_ip)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_faceid_ip)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])

    faceid_embeds_faceid_ip = face_extractor(img_faceid_ip)

    final_image = []

    for i in range (num_prompt_faceid_ip):
        if (is_xlturbo_faceid_ip == True) :
            image = pipe_faceid_ip(
                image_embeds=faceid_embeds_faceid_ip,
                prompt=prompt_faceid_ip,
                num_images_per_prompt=num_images_per_prompt_faceid_ip,
                guidance_scale=guidance_scale_faceid_ip,
                strength=denoising_strength_faceid_ip,
                num_inference_steps=num_inference_step_faceid_ip,
                height=height_faceid_ip,
                width=width_faceid_ip,
                generator = generator,
                callback_on_step_end=check_faceid_ip, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images
        elif (is_xl_faceid_ip == True) :
            image = pipe_faceid_ip(
                image_embeds=faceid_embeds_faceid_ip,
                prompt=prompt_faceid_ip,
                negative_prompt=negative_prompt_faceid_ip,
#                prompt_embeds=conditioning,
#                pooled_prompt_embeds=pooled,
#                negative_prompt_embeds=neg_conditioning,
#                negative_pooled_prompt_embeds=neg_pooled,
                num_images_per_prompt=num_images_per_prompt_faceid_ip,
                guidance_scale=guidance_scale_faceid_ip,
                strength=denoising_strength_faceid_ip,
                num_inference_steps=num_inference_step_faceid_ip,
                height=height_faceid_ip,
                width=width_faceid_ip,
                generator = generator,
                callback_on_step_end=check_faceid_ip, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images            
        else : 
            image = pipe_faceid_ip(
                image_embeds=faceid_embeds_faceid_ip,
                prompt_embeds=conditioning,
                negative_prompt_embeds=neg_conditioning,
                num_images_per_prompt=num_images_per_prompt_faceid_ip,
                guidance_scale=guidance_scale_faceid_ip,
                strength=denoising_strength_faceid_ip,
                num_inference_steps=num_inference_step_faceid_ip,
                height=height_faceid_ip,
                width=width_faceid_ip,
                generator = generator,
                callback_on_step_end=check_faceid_ip, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images        

        for j in range(len(image)):
            savename = f"outputs/{timestamper()}.png"
            if use_gfpgan_faceid_ip == True :
                image[j] = image_gfpgan_mini(image[j])             
            image[j].save(savename)
            final_image.append(savename)

    print(f">>>[IP-Adapter 🖌️ ]: generated {num_prompt_faceid_ip} batch(es) of {num_images_per_prompt_faceid_ip}")
    reporting_faceid_ip = f">>>[IP-Adapter 🖌️ ]: "+\
        f"Settings : Model={modelid_faceid_ip} | "+\
        f"XL model={is_xl_faceid_ip} | "+\
        f"Sampler={sampler_faceid_ip} | "+\
        f"Steps={num_inference_step_faceid_ip} | "+\
        f"CFG scale={guidance_scale_faceid_ip} | "+\
        f"Size={width_faceid_ip}x{height_faceid_ip} | "+\
        f"GFPGAN={use_gfpgan_faceid_ip} | "+\
        f"Token merging={tkme_faceid_ip} | "+\
        f"LoRA model={lora_model_faceid_ip} | "+\
        f"LoRA weight={lora_weight_faceid_ip} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Denoising strength={denoising_strength_faceid_ip} | "+\
        f"Prompt={prompt_faceid_ip} | "+\
        f"Negative prompt={negative_prompt_faceid_ip}"
    print(reporting_faceid_ip)         

    exif_writer_png(reporting_faceid_ip, final_image)

    del nsfw_filter_final, feat_ex, pipe_faceid_ip, generator, faceid_embeds_faceid_ip, compel, conditioning, neg_conditioning, image
    clean_ram()

    print(f">>>[IP-Adapter 🖌️ ]: leaving module")
    return final_image, final_image 
