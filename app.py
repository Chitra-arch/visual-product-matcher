import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from utils.image_utils import preprocess_image, load_product_embeddings, find_similar_products

# App title and layout
st.set_page_config(page_title="Visual Product Matcher", layout="centered")
st.title("🕶️ Visual Product Matcher")
st.markdown("Match any product visually using CLIP embeddings and smart filters ✨")

# Gradient background + zoom animation
st.markdown("""
<style>
body {
  background: linear-gradient(to right, #ffecd2, #fcb69f);
  background-attachment: fixed;
}

.product-card {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  background-color: #ffffffdd;
  padding: 20px;
  border-radius: 16px;
  margin-bottom: 24px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.1);
}

.product-card img {
  width: 220px;
  height: auto;
  border-radius: 12px;
  margin-right: 24px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
  animation: zoomInOut 4s ease-in-out infinite alternate;
}

@keyframes zoomInOut {
  0% { transform: scale(1); }
  100% { transform: scale(1.1); }
}

.product-details {
  color: #000000;
}

.product-details h4 {
  margin: 0;
  font-size: 22px;
}

.product-details p {
  margin: 6px 0;
  font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Load Lottie animation
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"
lottie_animation = load_lottie_url(lottie_url)
if lottie_animation:
    st_lottie(lottie_animation, height=250)

# Load image from URL robustly
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except (requests.RequestException, UnidentifiedImageError):
        return None

# Upload or URL input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
image_url = st.text_input("Or paste an image URL")

# Preview image and handle errors
image = None
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    except UnidentifiedImageError:
        st.error("⚠️ Unable to read the uploaded image. Please try another file.")
elif image_url:
    image = load_image_from_url(image_url)
    if image:
        st.image(image, caption="Image from URL", use_container_width=True)
    else:
        st.error("⚠️ Invalid image URL or failed to load image.")

# Category selector
category = st.selectbox("Choose a category", ["All", "Shoes", "Bags", "Clothing", "Electronics", "Fitness", "Home", "Accessories"])

# Similarity score filter
min_score = st.slider("Minimum similarity score", 0.0, 1.0, 0.5, step=0.05)

# Cache embeddings for performance
@st.cache_data
def get_embeddings():
    return load_product_embeddings()

product_data, product_embeddings = get_embeddings()

# Run similarity search
if image:
    with st.spinner("🔍 Matching product..."):
        try:
            image_tensor = preprocess_image(image)
            results = find_similar_products(image_tensor, product_embeddings, product_data, category=category)
            results = [r for r in results if r.get('score', 1.0) >= min_score]
            st.subheader("🔍 Most Similar Product")
            for item in results:
                st.markdown(f"""
                    <div class="product-card">
                        <img src="{item['image_url']}">
                        <div class="product-details">
                            <h4>{item['name']}</h4>
                            <p>{item['description']}</p>
                            <p><strong>Category:</strong> {item['category']}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Failed to process image: {e}")
