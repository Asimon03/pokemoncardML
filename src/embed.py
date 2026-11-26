import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Takes a list of images and returns an array of embeddings with shape (len(images), 512)
def embed_images(images, batch_size=32):
    embeddings = []

	# Processes images in batches
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]

		#Load every image in this batch 
        loaded = []
        for item in batch:
            if isinstance(item, Image.Image):
                loaded.append(item.convert("RGB"))
            else:
                with Image.open(item) as im:
                    loaded.append(im.convert("RGB"))
		# Resizing to match what CLIP wants
        inputs = processor(images=loaded, return_tensors="pt").to(device)

		# Run the images through CLIPs encoder
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            emb = model.visual_projection(vision_out.pooler_output)
		# Normalize
        emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0).astype(np.float32)