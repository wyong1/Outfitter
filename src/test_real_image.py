import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image

# === CONFIG ===
MODEL_PATH = "models/deepfashion_grouped_classifier.keras" 
DATA_DIR = "data/fashion_grouped/train" 
IMG_SIZE = (224, 224)

def main():
    img_path = input("Enter path to image you want to test: ").strip()
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return

    print("Loading model...")
    model = load_model(MODEL_PATH)

    # Get class names from folder structure (already human-readable like 'Tops', 'Jackets', etc.)
    class_names = sorted(next(os.walk(DATA_DIR))[1])

    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_batch)[0]

    # Get top 3 predictions
    top_indices = np.argsort(pred)[::-1][:3]
    top_classes = [class_names[i] for i in top_indices]
    top_probs = [pred[i] for i in top_indices]

    print("\nTop 3 predictions:")
    for i in range(3):
        print(f"{i+1}. {top_classes[i]} ({top_probs[i]*100:.2f}%)")

    # Show image + top 3 predictions
    labels = [f"{cls} ({prob*100:.1f}%)" for cls, prob in zip(top_classes, top_probs)]
    title_text = "Top-3:\n" + "\n".join([f"{i+1}. {label}" for i, label in enumerate(labels)])

    plt.imshow(img)
    plt.axis('off')
    plt.title(title_text)
    plt.show()

if __name__ == "__main__":
    main()
