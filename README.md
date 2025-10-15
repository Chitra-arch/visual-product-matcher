# 🧠 Visual Product Matcher

A web application that helps users find visually similar products based on uploaded images. Built for a technical assessment, this app showcases image embedding, similarity search, and elegant UI design.

## 🚀 Features

- 📤 Upload image or paste URL
- 🔍 View visually similar products with scores
- 🎛️ Filter results by similarity score
- 📱 Mobile responsive layout
- ✨ Onboarding animations and elegant color palette
- 🧯 Error handling and loading states

## 🛠 Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Frontend    | Streamlit + HTML/CSS animations     |
| Backend     | Python + CLIP (OpenAI)              |
| Hosting     | Hugging Face Spaces (free tier)     |
| Dataset     | Public product images (50+ entries) |

## 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/yourusername/visual-product-matcher.git
cd visual-product-matcher

# Create virtual environment
python -m venv vpm-env
source vpm-env/bin/activate  # Windows: vpm-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate product embeddings
python generate_embeddings.py

# Run the app
streamlit run app.py
