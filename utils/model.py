# model.py — CLIP similarity logic

from transformers import CLIPModel, CLIPProcessor
import torch

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Compare query image with product embeddings
def get_similar_products(query_embedding, product_embeddings, products_df):
    scores = torch.nn.functional.cosine_similarity(query_embedding, product_embeddings)
    products_df['score'] = scores.tolist()
    return products_df.sort_values(by='score', ascending=False)
