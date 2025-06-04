# main attack script placeholder
import os
import torch
from transformers import AutoTokenizer
from llava.model import LlavaLlamaForCausalLM
from watermark_vocab import get_green_red_vocab
from watermark_sampling import apply_watermark_bias
from watermark_detector import detect_green_ratio
from extract_hidden import extract_hidden_features
from compute_nullspace import compute_nullspace_basis
from apply_projection import project_hidden  # 实现为 hook function

# ==== 配置参数 ====
image_path = "data/coco_sample/0000001.jpg"
prompt = "Describe what's in the image."
blurred_image_path = "data/coco_sample/0000001_blur.jpg"
model_name = "liuhaotian/llava-v1.5-7b"
layer_idx = 10
gamma = 2.0
k = 16

# ==== 加载模型和 tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlavaLlamaForCausalLM.from_pretrained(model_name).cuda().eval()

# ==== 构建 green token vocab ====
green_vocab, _ = get_green_red_vocab(tokenizer, ratio=0.5)

# ==== 原始生成（有水印） ====
def generate_with_watermark(prompt, image_tensor):
    inputs = model.processor(images=image_tensor, text=prompt, return_tensors="pt").to("cuda")
    past = None
    output_tokens = []

    for _ in range(50):
        if output_tokens:
            input_ids = torch.tensor([[output_tokens[-1]]]).to("cuda")
        else:
            input_ids = inputs.input_ids

        out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        logits = apply_watermark_bias(logits, tokenizer, green_vocab, gamma)
        next_token = torch.argmax(logits, dim=-1)
        output_tokens.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(output_tokens, skip_special_tokens=True)

# ==== 提取正负特征 ====
extract_hidden_features(image_path, prompt, "features/pos.pt", layer_idx)
extract_hidden_features(blurred_image_path, prompt, "features/neg.pt", layer_idx)

# ==== 计算 null space 基方向 ====
X_pos = torch.load("features/pos.pt")
X_neg = torch.load("features/neg.pt")
E = X_pos - X_neg
V_k = compute_nullspace_basis(E, k=k)
torch.save(V_k, "features/nullspace.pt")

# ==== 注册 forward hook 攻击 ====
def hook_fn(module, input, output):
    out_proj = project_hidden(output, V_k)
    return out_proj

handle = model.model.layers[layer_idx].mlp.fc1.register_forward_hook(hook_fn)

# ==== 攻击后重新生成 ====
from PIL import Image
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img = Image.open(image_path).convert("RGB")
image_tensor = transform(img).unsqueeze(0).cuda()

print("\n🔰 Watermarked output:")
orig_text = generate_with_watermark(prompt, image_tensor)
print(orig_text)
ratio_orig = detect_green_ratio(orig_text, green_vocab)
print(f"✅ Green token ratio: {ratio_orig:.3f}")

print("\n💥 After NullWM attack:")
handle.remove()  # 防止叠加
model = LlavaLlamaForCausalLM.from_pretrained(model_name).cuda().eval()
handle = model.model.layers[layer_idx].mlp.fc1.register_forward_hook(hook_fn)
attacked_text = generate_with_watermark(prompt, image_tensor)
print(attacked_text)
ratio_attacked = detect_green_ratio(attacked_text, green_vocab)
print(f"❌ Green token ratio after attack: {ratio_attacked:.3f}")
