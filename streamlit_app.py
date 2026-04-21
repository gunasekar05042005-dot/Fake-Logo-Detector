import streamlit as st
import torch
from PIL import Image
import clip

# Load the CLIP model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32', device=device)

st.title('Logo Detection with CLIP and Streamlit')

# Upload an image
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png'])

if uploaded_file is not None:
    # Preprocess the image
    image = Image.open(uploaded_file)
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Define logo labels
    logo_labels = ['Nike', 'Adidas', 'Puma', 'Pepsi', 'Coca Cola']
    text_inputs = torch.cat([clip.tokenize(logo) for logo in logo_labels]).to(device)
    
    # Compute features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        
    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    best_label = logo_labels[similarity.argmax().item()]
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted Logo: {best_label}')
