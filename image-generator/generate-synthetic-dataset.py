from diffusers import StableDiffusionPipeline
import torch
import os
from tqdm import tqdm

seed = 0
num_per_prompt = 5
num_images_to_generate = int(100000 / num_per_prompt)
model_path = # path to generated LoRA parameters
wm_image_save_path = # destination path to WM class generated images
nm_image_save_path = # destination path to NM class generated images
tracking_file = # path to the image count tracking file

with open(tracking_file, 'r') as f:
    image_num = int(f.read())

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")
pipe.load_lora_weights(model_path)

generator = torch.Generator("cuda").manual_seed(seed)
images = []


for i in tqdm(range(num_images_to_generate)):

    if i < num_images_to_generate / 2:
        prompt = "Bone surface modification with meat."
        images = pipe(prompt, generator = generator, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=num_per_prompt).images[:]

        for image in images:
            image.save(os.path.join(wm_image_save_path, f'wm_{image_num}.png'))
            image_num += 1

        with open(tracking_file, 'w') as f:
            f.write(str(image_num))

    else:
        prompt = "Bone surface modification without meat."
        images = pipe(prompt, generator = generator, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=num_per_prompt).images[:]

        for image in images:
            image.save(os.path.join(nm_image_save_path, f'nm_{image_num}.png'))
            image_num += 1

        with open(tracking_file, 'w') as f:
            f.write(str(image_num))