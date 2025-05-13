import os
import random
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# CONFIG 
MODEL_PATH = "models/deepfashion_grouped_classifier.keras" 
DATA_DIR = "data/fashion_grouped/train"
IMG_SIZE = (224, 224)

# Recommendations for grouped categories
category_matches = {
    "Tops": ["Pants", "Jackets", "Athleisure"],
    "Sweaters": ["Pants", "Jackets"],
    "Jackets": ["Tops", "Sweaters", "Pants"],
    "Pants": ["Tops", "Sweaters"],
    "Shorts": ["Tops", "Outerwear"],
    "Dresses": ["Outerwear", "Jackets"],
    "Skirts": ["Tops", "Outerwear"],
    "Outerwear": ["Tops", "Pants"],
    "Jumpsuits": ["Outerwear", "Jackets"],
    "Athleisure": ["Tops", "Sweaters"]
}

# Only load model once per session
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

def predict_top3(image_path):
    model = get_model()
    folders = sorted(os.listdir(DATA_DIR))   # folder names = class names
    index_to_label = {i: folder for i, folder in enumerate(folders)}

    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_batch)[0]
    top_indices = np.argsort(preds)[::-1][:3]
    top_labels = [index_to_label[idx] for idx in top_indices]
    confidences = [preds[idx] for idx in top_indices]
    return list(zip(top_labels, confidences))

def recommend_outfit(main_category):
    outfit = []
    recommendations = category_matches.get(main_category, [])

    for category in recommendations:
        folder_path = os.path.join(DATA_DIR, category)
        if os.path.exists(folder_path):
            candidates = os.listdir(folder_path)
            if candidates:
                outfit.append(os.path.join(folder_path, random.choice(candidates)))
    return outfit
