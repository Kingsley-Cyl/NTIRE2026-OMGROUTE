# import os
# import glob
# import torch
# from PIL import Image
# from torchvision import transforms
# import torchvision.transforms.functional as F
# from transformers import AutoTokenizer, CLIPTextModel
# from diffusers.training_utils import free_memory

# # 从你未修改的 OMGSR 源码中直接导入
# from .infer.omgsr_s_infer_model import OMGSR_S_Infer
# from .infer.wavelet_color_fix import adain_color_fix

# def omgsr_inference(model_dir, input_path, output_path, device):
#     """
#     符合 NTIRE 官方要求的 4 参数推理接口
#     针对 omgsr_s_512 版本
#     """
#     # ====== 超参数设置 (对应 omgsr_s_512.yml) ======
#     mid_timestep = 273
#     process_size = 512
#     upscale = 4
#     weight_dtype = torch.float16  # 移动端推荐使用半精度
    
#     # 路径设置
#     # 注意: 如果比赛在离线环境评测，建议将 stable-diffusion-2-1-base 也下载到 model_dir 中，
#     # 并将此处的 sd_path 指向本地路径，例如 os.path.join(model_dir, 'sd-2-1-base')
#     sd_path = "/home/jiacheng/Cyl/stable-diffusion-2-1-base" 
    
#     # 我们假设你的 LoRA 权重放在了官方规定的 model_zoo/teamXX_OMGSR/lora 目录下
#     lora_path = os.path.join(model_dir, "lora") 

#     # ====== 1. 编码 Prompt (完全复用你原来的逻辑) ======
#     tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
#     text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(device, dtype=weight_dtype)
    
#     with torch.no_grad():
#         # s_512 版本默认空 prompt
#         prompt_embeds = text_encoder(
#             tokenizer(
#                 "", 
#                 max_length=tokenizer.model_max_length,
#                 padding="max_length",
#                 truncation=True,
#                 return_tensors="pt",
#             ).input_ids.to(device)
#         )[0]#.unsqueeze(0)
        
#     del tokenizer
#     del text_encoder
#     free_memory()

#     # ====== 2. 初始化 OMGSR 模型 ======
#     net_sr = OMGSR_S_Infer(
#         sd_path=sd_path,
#         lora_path=lora_path,
#         mid_timestep=mid_timestep,
#         device=device,
#         weight_dtype=weight_dtype
#     )
    
#     # ====== 3. 开始批量推理 ======
#     os.makedirs(output_path, exist_ok=True)
#     image_names = sorted(glob.glob(os.path.join(input_path, "*.png")) + 
#                          glob.glob(os.path.join(input_path, "*.jpg")) + 
#                          glob.glob(os.path.join(input_path, "*.jpeg")))
    
#     tile_size = process_size // 8
#     tile_overlap = tile_size // 2

#     for image_name in image_names:
#         input_image = Image.open(image_name).convert('RGB')
#         ori_width, ori_height = input_image.size
#         resize_flag = False

#         if ori_width < process_size // upscale or ori_height < process_size // upscale:
#             scale = (process_size // upscale) / min(ori_width, ori_height)
#             input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
#             resize_flag = True

#         input_image = input_image.resize((input_image.size[0] * upscale, input_image.size[1] * upscale))
#         new_width = input_image.width - input_image.width % 8
#         new_height = input_image.height - input_image.height % 8
#         input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        
#         bname = os.path.basename(image_name).split('.')[0] + ".png"

#         with torch.no_grad():
#             lq_img = F.to_tensor(input_image).unsqueeze(0).to(device=device, dtype=weight_dtype) * 2 - 1
#             output_image, _ = net_sr(lq_img, prompt_embeds, tile_size, tile_overlap)

#         output_image = output_image * 0.5 + 0.5
#         output_image = torch.clip(output_image, 0, 1).float()
#         output_pil = transforms.ToPILImage()(output_image[0].cpu())

#         output_pil = adain_color_fix(target=output_pil, source=input_image)

#         target_width, target_height = ori_width * upscale, ori_height * upscale
#         if resize_flag or output_pil.size != (target_width, target_height):
#             output_pil = output_pil.resize((target_width, target_height), Image.LANCZOS)
            
#         output_pil.save(os.path.join(output_path, bname))
#         print(f"Processed: {bname}")

import os
import glob
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm

# 导入你修改后的多 LoRA 模型结构
from .infer.omgsr_s_infer_model_multi_lora import OMGSR_S_Infer
from .infer.wavelet_color_fix import adain_color_fix

def omgsr_inference(model_dir, input_path, output_path, device):
    """
    符合 NTIRE 官方要求的 4 参数推理接口 (Multi-LoRA 动态切换版本)
    """
    process_size = 512
    upscale = 4
    mid_timestep = 273
    weight_dtype = torch.float16
    align_method = 'adain'

    # sd_path = os.path.join(model_dir, "sd-2-1-base")
    sd_path = '/home/jiacheng/Cyl/stable-diffusion-2-1-base'
    lora_path_rect = os.path.join(model_dir, "lora_rect")
    lora_path_square = os.path.join(model_dir, "lora_square")
    embeds_path = os.path.join(model_dir, "empty_embeds.pt")

    if not os.path.exists(embeds_path):
        raise FileNotFoundError(f"Cannot find {embeds_path}! Please ensure it is in your model_zoo folder.")
        
    prompt_embeds = torch.load(embeds_path).to(device, dtype=weight_dtype)
    print("Successfully loaded pre-computed empty prompt embeddings.")

    net_sr = OMGSR_S_Infer(
        sd_path=sd_path,
        lora_path_rect=lora_path_rect,
        lora_path_square=lora_path_square,
        mid_timestep=mid_timestep,
        device=device,
        weight_dtype=weight_dtype
    )

    os.makedirs(output_path, exist_ok=True)
    image_names = sorted(glob.glob(os.path.join(input_path, "*.png")) + 
                         glob.glob(os.path.join(input_path, "*.jpg")) + 
                         glob.glob(os.path.join(input_path, "*.jpeg")))

    square_images = []
    rect_images = []
    for image_name in image_names:
        with Image.open(image_name) as img:
            if img.size[0] == img.size[1]:
                square_images.append(image_name)
            else:
                rect_images.append(image_name)
                
    print(f"Total images: {len(image_names)} (Square: {len(square_images)}, Rect: {len(rect_images)})")

    def process_image_group(img_list, is_square_group):
        if not img_list:
            return

        # 动态合并对应的 LoRA 以加速推理
        net_sr.merge_current_lora(is_square=is_square_group)
        
        group_name = "Square" if is_square_group else "Rect"
        for image_name in tqdm(img_list, desc=f"Processing {group_name} Images"):
            input_image = Image.open(image_name).convert('RGB')
            ori_width, ori_height = input_image.size
            
            rscale = upscale
            resize_flag = False

            if ori_width < process_size // rscale or ori_height < process_size // rscale:
                scale = (process_size // rscale) / min(ori_width, ori_height)
                input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
                resize_flag = True

            input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
            new_width = input_image.width - input_image.width % 8
            new_height = input_image.height - input_image.height % 8
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
            bname = os.path.basename(image_name).split('.')[0] + ".png"
            
            tile_size = process_size // 8
            tile_overlap = tile_size // 2

            with torch.no_grad():
                lq_img = F.to_tensor(input_image).unsqueeze(0).to(device=device, dtype=weight_dtype) * 2 - 1
                output_image, _ = net_sr(lq_img, prompt_embeds, tile_size, tile_overlap)

            output_image = output_image * 0.5 + 0.5
            output_image = torch.clip(output_image, 0, 1).float()
            output_pil = transforms.ToPILImage()(output_image[0].cpu())

            if align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            
            # 最后强制矫正尺寸
            target_width = ori_width * rscale
            target_height = ori_height * rscale
            if resize_flag or output_pil.size != (target_width, target_height):
                output_pil = output_pil.resize((target_width, target_height), Image.LANCZOS)
                
            output_pil.save(os.path.join(output_path, bname))
            
        net_sr.unmerge_current_lora()

    process_image_group(square_images, is_square_group=True)
    process_image_group(rect_images, is_square_group=False)