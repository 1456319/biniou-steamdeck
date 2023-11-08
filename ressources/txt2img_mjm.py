# https://github.com/Woolverine94/biniou
# txt2img_mjm.py
import gradio as gr
import os
from diffusers import DiffusionPipeline
from compel import Compel, ReturnedEmbeddingsType
import torch
import time
import random
from ressources.scheduler import *
from ressources.gfpgan import *
import tomesd

device_txt2img_mjm = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gestion des modèles
model_path_txt2img_mjm = "./models/Midjourney_mini/"
model_path_txt2img_mjm_safetychecker = "./models/Stable_Diffusion/" 
os.makedirs(model_path_txt2img_mjm, exist_ok=True)
model_list_txt2img_mjm = []

for filename in os.listdir(model_path_txt2img_mjm):
    f = os.path.join(model_path_txt2img_mjm, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_txt2img_mjm.append(f)

model_list_txt2img_mjm_builtin = [
    "openskyml/midjourney-mini",
]

for k in range(len(model_list_txt2img_mjm_builtin)):
    model_list_txt2img_mjm.append(model_list_txt2img_mjm_builtin[k])

# Bouton Cancel
stop_txt2img_mjm = False

def initiate_stop_txt2img_mjm() :
    global stop_txt2img_mjm
    stop_txt2img_mjm = True

def check_txt2img_mjm(step, timestep, latents) :
    global stop_txt2img_mjm
    if stop_txt2img_mjm == False :
#        result_preview = preview_image(step, timestep, latents, pipe_txt2img_mjm)
        return
    elif stop_txt2img_mjm == True :
        stop_txt2img_mjm = False
        try:
            del ressources.txt2img_mjm.pipe_txt2img_mjm
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def image_txt2img_mjm(
    modelid_txt2img_mjm,
    sampler_txt2img_mjm,
    prompt_txt2img_mjm,
    negative_prompt_txt2img_mjm,
    num_images_per_prompt_txt2img_mjm,
    num_prompt_txt2img_mjm,
    guidance_scale_txt2img_mjm,
    num_inference_step_txt2img_mjm,
    height_txt2img_mjm,
    width_txt2img_mjm,
    seed_txt2img_mjm,
    use_gfpgan_txt2img_mjm,
    nsfw_filter,
    tkme_txt2img_mjm,
    progress_txt2img_mjm=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Midjourney-mini 🖼️ ]: starting module")
    
#    global pipe_txt2img_mjm
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_txt2img_mjm_safetychecker, device_txt2img_mjm, nsfw_filter)

    if modelid_txt2img_mjm[0:9] == "./models/" :
        pipe_txt2img_mjm = DiffusionPipeline.from_single_file(
            modelid_txt2img_mjm, 
            torch_dtype=torch.float32, 
#            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
        )
    else :        
        pipe_txt2img_mjm = DiffusionPipeline.from_pretrained(
            modelid_txt2img_mjm, 
            cache_dir=model_path_txt2img_mjm, 
            torch_dtype=torch.float32, 
#            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
    
    pipe_txt2img_mjm = get_scheduler(pipe=pipe_txt2img_mjm, scheduler=sampler_txt2img_mjm)
    pipe_txt2img_mjm = pipe_txt2img_mjm.to(device_txt2img_mjm)
    pipe_txt2img_mjm.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_txt2img_mjm, ratio=tkme_txt2img_mjm)
    
    if seed_txt2img_mjm == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_txt2img_mjm
    generator = []
    for k in range(num_prompt_txt2img_mjm):
        generator.append([torch.Generator(device_txt2img_mjm).manual_seed(final_seed + (k*num_images_per_prompt_txt2img_mjm) + l ) for l in range(num_images_per_prompt_txt2img_mjm)])

    prompt_txt2img_mjm = str(prompt_txt2img_mjm)
    negative_prompt_txt2img_mjm = str(negative_prompt_txt2img_mjm) 
    if prompt_txt2img_mjm == "None":
        prompt_txt2img_mjm = ""
    if negative_prompt_txt2img_mjm == "None":
        negative_prompt_txt2img_mjm = ""
        
    compel = Compel(tokenizer=pipe_txt2img_mjm.tokenizer, text_encoder=pipe_txt2img_mjm.text_encoder, truncate_long_prompts=False)
    conditioning = compel.build_conditioning_tensor(prompt_txt2img_mjm)
    neg_conditioning = compel.build_conditioning_tensor(negative_prompt_txt2img_mjm)    
    [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])   
    
    final_image = []
    for i in range (num_prompt_txt2img_mjm):
        image = pipe_txt2img_mjm(
            prompt_embeds=conditioning,
            negative_prompt_embeds=neg_conditioning,
            height=height_txt2img_mjm,
            width=width_txt2img_mjm,
            num_images_per_prompt=num_images_per_prompt_txt2img_mjm,
            num_inference_steps=num_inference_step_txt2img_mjm,
            guidance_scale=guidance_scale_txt2img_mjm,
            generator = generator[i],
            callback = check_txt2img_mjm, 
        ).images

        final_seed = []
        for j in range(len(image)):
            timestamp = time.time()
            seed_id = random_seed + i*num_images_per_prompt_txt2img_mjm + j if (seed_txt2img_mjm == 0) else seed_txt2img_mjm + i*num_images_per_prompt_txt2img_mjm + j
            savename = f"outputs/{seed_id}_{timestamp}.png"
            if use_gfpgan_txt2img_mjm == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(savename)
            final_seed.append(seed_id)

    print(f">>>[Midjourney-mini 🖼️ ]: generated {num_prompt_txt2img_mjm} batch(es) of {num_images_per_prompt_txt2img_mjm}")
    reporting_txt2img_mjm = f">>>[Midjourney-mini 🖼️ ]: "+\
        f"Settings : Model={modelid_txt2img_mjm} | "+\
        f"Sampler={sampler_txt2img_mjm} | "+\
        f"Steps={num_inference_step_txt2img_mjm} | "+\
        f"CFG scale={guidance_scale_txt2img_mjm} | "+\
        f"Size={width_txt2img_mjm}x{height_txt2img_mjm} | "+\
        f"GFPGAN={use_gfpgan_txt2img_mjm} | "+\
        f"Token merging={tkme_txt2img_mjm} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Prompt={prompt_txt2img_mjm} | "+\
        f"Negative prompt={negative_prompt_txt2img_mjm} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_txt2img_mjm) 

    del nsfw_filter_final, feat_ex, pipe_txt2img_mjm, generator, compel, conditioning, neg_conditioning, image
    clean_ram()

    print(f">>>[Midjourney-mini 🖼️ ]: leaving module") 
    return final_image, final_image
