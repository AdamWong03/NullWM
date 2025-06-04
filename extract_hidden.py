# extract hidden states from model placeholder
import torch
from PIL import Image
from torchvision import transforms
from llava.model import LlavaLlamaForCausalLM

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_hidden_features(image_path, prompt, save_path, layer_idx=10):
    model = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b").cuda().eval()
    tokenizer = model.tokenizer
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()
    inputs = model.processor(images=image, text=prompt, return_tensors="pt").to("cuda")

    def hook(module, input, output):
        torch.save(output.mean(dim=1).detach().cpu(), save_path)

    handle = model.model.layers[layer_idx].mlp.fc1.register_forward_hook(hook)
    _ = model.generate(**inputs, max_new_tokens=20)
    handle.remove()
