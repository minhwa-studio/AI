import subprocess
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login


# 허깅페이스 로그인
login(token="")

# ------------------------------------------------
# LoRA 학습
# ------------------------------------------------
print("\nLoRA 학습 스크립트를 다운로드합니다...")
# '-k' 또는 '--insecure' 옵션을 추가하여 인증서 오류를 무시합니다.
subprocess.run(["curl", "-O", "-k", "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora.py"], check=True)
print("스크립트 다운로드 완료. ✅")

print("\n스타일 학습을 시작합니다... 🚀")
try:
    subprocess.run([
        "accelerate", "launch",
        "--mixed_precision=bf16",
        "train_dreambooth_lora.py",
        "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
        "--instance_data_dir=./SDv1.5/minhwa",
        "--output_dir=./sd15_lora_minhwa",
        "--instance_prompt", 'a painting of minhwa, a traditional Korean painting, soft watercolor style, subtle colors, delicate brushstrokes',
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=1",
        "--learning_rate=1e-4",
        "--lr_scheduler=cosine",
        "--lr_warmup_steps=0",
        "--max_train_steps=1000",
        "--checkpointing_steps=200",
        "--seed=42",
        "--gradient_checkpointing"
    ], check=True)
    print("\n학습이 성공적으로 완료 ✅")

except FileNotFoundError:
    print("❌ 오류: 'accelerate' 또는 'train_dreambooth_lora.py'를 찾을 수 없습니다. ")
    exit(1)
except subprocess.CalledProcessError as e:
    print(f"❌ 학습 중 오류가 발생했습니다: {e}")
    exit(1)