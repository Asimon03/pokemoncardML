import torch
import cv2
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

img = Image.open("test_images/testimg.png").convert("RGB")
inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
    emb = model.visual_projection(vision_out.pooler_output)
emb = emb / emb.norm(dim=-1, keepdim=True)

print(emb.shape, "\nOpenCV", cv2.__version__)