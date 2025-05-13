# src/streamlit_app.py

import streamlit as st
import os
import matplotlib.pyplot as plt
from PIL import Image
from function_app import predict_top3, recommend_outfit

st.title("👗 Outfitter")

img_file = st.file_uploader("Upload a clothing image", type=["jpg", "jpeg", "png"])

if img_file:
    # Save uploaded file
    img_path = os.path.join("temp_uploaded.jpg")
    with open(img_path, "wb") as f:
        f.write(img_file.getbuffer())

    img = Image.open(img_path)
    #st.image(img, caption="Uploaded Image", use_container_width=True)


    # Predict top 3
    predictions = predict_top3(img_path)
    st.subheader("What item is this?")
    options = [f"{label} ({confidence*100:.1f}%)" for label, confidence in predictions]
    selected = st.radio("Choose best match:", options)
    main_category = predictions[options.index(selected)][0]

    # Recommend outfit
    outfit_paths = recommend_outfit(main_category)
    st.subheader("Suggested outfit pieces:")

    cols = st.columns(len(outfit_paths) + 1)
    cols[0].image(img_path, caption=main_category, use_container_width=True)

    for i, path in enumerate(outfit_paths):
        with cols[i+1]:
            outfit_img = Image.open(path)
            label = os.path.basename(os.path.dirname(path))
            st.image(outfit_img, caption=label, use_container_width=True)
