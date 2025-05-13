import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# CONFIG
MODEL_PATH = "models/deepfashion_grouped_classifier.keras"     
DATA_DIR = "data/fashion_grouped/train"                     
IMG_SIZE = (224, 224)

# CATEGORY GROUPS
category_groups = [
    "Jackets", "Tops", "Sweaters", "Athleisure", "Outerwear",
    "Pants", "Skirts", "Shorts", "Dresses", "Jumpsuits"
]

# MATCHING RULES between groups
group_matches = {
    "Tops": ["Pants", "Skirts", "Shorts", "Jackets"],
    "Sweaters": ["Pants", "Skirts", "Jackets"],
    "Athleisure": ["Athleisure", "Pants", "Shorts"],
    "Jackets": ["Tops", "Sweaters", "Dresses"],
    "Outerwear": ["Tops", "Sweaters", "Dresses"],
    "Pants": ["Tops", "Sweaters"],
    "Skirts": ["Tops", "Sweaters"],
    "Shorts": ["Tops", "Athleisure"],
    "Dresses": ["Jackets", "Outerwear"],
    "Jumpsuits": ["Jackets", "Outerwear"]
}

def recommend_outfit(image_path):
    model = load_model(MODEL_PATH)

    folders = sorted(os.listdir(DATA_DIR))  # ['Athleisure', 'Dresses', etc.]
    index_to_group = {i: folder for i, folder in enumerate(folders)}

    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_batch)[0]
    top3_indices = np.argsort(preds)[::-1][:3]

    outfit = [image_path]
    used_types = set()

    for idx in top3_indices:
        main_group = index_to_group[idx]
        if main_group not in group_matches:
            continue

        print(f"\nTrying main item: {main_group}")
        used_types.add(main_group)

        recommendations = group_matches.get(main_group, [])

        found = False
        for rec_group in recommendations:
            if rec_group not in used_types and rec_group in folders:
                folder_path = os.path.join(DATA_DIR, rec_group)
                candidates = os.listdir(folder_path)
                if candidates:
                    outfit.append(os.path.join(folder_path, random.choice(candidates)))
                    used_types.add(rec_group)
                    found = True

        if found:
            print(f"Outfit generated with: {main_group}")
            break
        else:
            print(f"No recommendations for {main_group}")

    print(f"\nRecommending {len(outfit)-1} matching pieces:")
    for path in outfit[1:]:
        print(f"- {os.path.basename(os.path.dirname(path))} / {os.path.basename(path)}")

    # Display results
    fig, axes = plt.subplots(1, len(outfit), figsize=(5 * len(outfit), 5))
    if len(outfit) == 1:
        axes = [axes]
    for ax, img_path in zip(axes, outfit):
        img_disp = image.load_img(img_path, target_size=IMG_SIZE)
        ax.imshow(img_disp)
        ax.axis('off')
        ax.set_title(os.path.basename(img_path))
    plt.show()

if __name__ == "__main__":
    while True:
        img_path = input("\nEnter path to clothing image (or 'q' to quit): ").strip()
        if img_path.lower() == 'q':
            print("Exiting Outfit Recommender. Goodbye!")
            break
        if os.path.isfile(img_path):
            recommend_outfit(img_path)
            again = input("\nWould you like to recommend another image? (y/n): ").strip().lower()
            if again != 'y':
                print("Exiting Outfit Recommender. Goodbye!")
                break
        else:
            print("File not found. Please try again.")
