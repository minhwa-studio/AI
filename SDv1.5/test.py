import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import cv2
import numpy as np
from diffusers.utils import load_image
from torchvision.utils import make_grid
from datetime import datetime
from torchvision.transforms import ToTensor, ToPILImage


# ------------------------------------------------
# 1. GPU 및 모델 로드
# ------------------------------------------------
print("GPU 및 모델을 로드합니다... ⏳")

device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
).to(device)

lora_path = "sd15_lora_minhwa"
pipeline.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")

# ------------------------------------------------
# 2. 이미지 전처리 및 스타일 변환
# ------------------------------------------------
print("\n테스트 이미지에 스타일을 적용합니다. 🖼️")

test_image_path = "./image/test3.jpg"
if not os.path.exists(test_image_path):
    print(f"❌ 오류: 테스트 이미지 '{test_image_path}'를 찾을 수 없습니다. 경로를 확인하세요.")
    exit(1)

init_image = load_image(test_image_path).convert("RGB")
init_image = init_image.resize((512, 512))

prompt = "a painting of minhwa, a traditional Korean painting, ink wash painting, soft watercolor style, subtle colors, delicate brushstrokes, traditional folk art, monochromatic"
negative_prompt = "japanese style, chinese style, ukiyo-e, sumi-e, western art, European style, masterpiece, oil painting, portrait, lowres, bad anatomy, bad hands, cropped, worst quality, deformed"

# 그리드 생성 및 주요 파라미터 조정
# [ strength 값 설명 ]
# - 0.5~0.7: 원본 구도를 유지하면서 상당 부분 변경 (추천 시작 값)
# - 0.8~0.9: 원본 구도를 많이 변경하면서 스타일 강하게 적용
strengths_to_test = [0.4,0.6,0.8]

# [ num_inference_steps 값 설명 ]
# - 10~20: 빠른 생성. 이미지가 거칠거나 덜 선명
# - 25~30: 일반적이고 균형 잡힌 품질 (기본값)
# - 40~50: 고품질. 생성 시간 증가
steps_to_test = [20,30,40]

# [ guidance_scale 값 설명 ]
# - 5~8: 프롬프트 영향력이 적고, 원본 모델의 스타일을 따름
# - 9~15: 프롬프트의 영향을 강하게 받아 원하는 스타일을 구현 (추천 시작 값)
scales_to_test = [9,11,13]

all_generated_images = []
generator = torch.Generator(device).manual_seed(100)

# 이미지 변환기 인스턴스 생성
transform_to_tensor = ToTensor()

print("다양한 파라미터 조합으로 이미지 생성 중...")
for s in strengths_to_test:
    for num_steps in steps_to_test:
        for g_scale in scales_to_test:
            print(f"-> strength: {s}, steps: {num_steps}, guidance_scale: {g_scale}")
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=s,
                num_inference_steps=num_steps,
                guidance_scale=g_scale,
                generator=generator
            )
            # PIL 이미지를 텐서로 변환 후 리스트에 추가
            tensor_image = transform_to_tensor(output.images[0])
            all_generated_images.append(tensor_image)

# 그리드 이미지 생성 및 저장
grid_image = make_grid(all_generated_images, nrow=len(scales_to_test) * len(steps_to_test), padding=5)

# 파일명 자동 생성
base_name = os.path.splitext(os.path.basename(test_image_path))[0]
output_dir = "./image/result"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_image_path = os.path.join(output_dir, f"{base_name}_params_grid.png")

# 텐서인 그리드 이미지를 다시 PIL 이미지로 변환하여 저장
to_pil_image = ToPILImage()
pil_grid_image = to_pil_image(grid_image)
pil_grid_image.save(output_image_path)

print(f"\n다양한 파라미터 조합으로 생성된 그리드 이미지가 '{output_image_path}'에 저장되었습니다. ✅")