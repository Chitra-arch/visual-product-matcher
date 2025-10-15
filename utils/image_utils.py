from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Convert image to CLIP embedding
def preprocess_image(image):
    if isinstance(image, str):
        image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
    elif hasattr(image, "read"):
        image = Image.open(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.squeeze(0)

# Load product metadata and embeddings
def load_product_embeddings():
    df = pd.read_csv("data/products.csv")
    embeddings = torch.load("data/product_embeddings.pt")
    return df, embeddings

# Find similar products using cosine similarity
def find_similar_products(image_tensor, product_embeddings, product_data, category="All"):
    image_embedding = image_tensor.detach().cpu().numpy().reshape(1, -1)

    if isinstance(product_embeddings, torch.Tensor):
        product_embeddings = product_embeddings.detach().cpu().numpy()

    similarities = cosine_similarity(image_embedding, product_embeddings)[0]

    results = []
    for idx, score in enumerate(similarities):
        product = product_data.iloc[idx]
        product_category = str(product["category"]).lower()
        if category.lower() == "all" or category.lower() in product_category:
            results.append({
                "name": product["name"],
                "description": product["description"],
                "image_url": product["image_url"],
                "category": product["category"],
                "score": score
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:1]
