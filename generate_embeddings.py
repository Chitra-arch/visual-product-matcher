# generate_embeddings.py — Embeds product images using CLIP
# Author: Chitra | MCA @ VIT

import pandas as pd
import torch
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load product metadata
df = pd.read_csv("data/products.csv")

embeddings = []

# Loop through each product
for idx, row in df.iterrows():
    image = None
    name = row['name']
    image_url = row.get('image_url', None)
    image_file = row.get('image', None)

    try:
        # Try loading from image_url first
        if pd.notna(image_url) and isinstance(image_url, str) and image_url.startswith("http"):
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")

        # Fallback to local image if URL fails or is missing
        elif pd.notna(image_file) and isinstance(image_file, str):
            local_path = Path("data") / image_file
            if local_path.exists():
                image = Image.open(local_path).convert("RGB")
            else:
                raise FileNotFoundError(f"Local image not found: {local_path}")

        # If image is loaded, embed it
        if image:
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_embedding = model.get_image_features(**inputs)
            embeddings.append(image_embedding.squeeze(0))
            print(f"[{idx+1}/{len(df)}] ✅ Embedded: {name}")
        else:
            raise ValueError("No valid image source found.")

    except Exception as e:
        print(f"[{idx+1}/{len(df)}] ❌ Failed: {name} — {e}")
        embeddings.append(torch.zeros(512))  # fallback zero vector

# Stack and save
embedding_tensor = torch.stack(embeddings)
torch.save(embedding_tensor, "data/product_embeddings.pt")
print("✅ All embeddings saved to data/product_embeddings.pt")
