import os
import cv2
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  #normalize
    return image

def preprocess_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for category in os.listdir(input_folder):
        category_path = os.path.join(input_folder, category)
        output_category_path = os.path.join(output_folder, category)

        os.makedirs(output_category_path, exist_ok=True)

        for file in os.listdir(category_path):
            img_path = os.path.join(category_path, file)
            processed_img = preprocess_image(img_path)

            #convert back to uint8 so we can save it
            processed_img = (processed_img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_category_path, file), processed_img)

if __name__ == "__main__":
    preprocess_folder('../data/raw', '../data/cleaned')
