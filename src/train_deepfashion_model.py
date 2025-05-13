import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing import image
import random
from PIL import Image

# CONFIG
DATA_DIR = "data/fashion_classifer/train"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
MODEL_PATH = "models/deepfashion_grouped_classifier.keras"

# Dataset exist
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset not found at {DATA_DIR}. Run your grouping script first!")  

# CLEAN DATASET
def clean_dataset(directory):
    removed = 0
    for root, dirs, files in os.walk(directory):
        for name in files:
            filepath = os.path.join(root, name)
            if os.path.islink(filepath) and not os.path.exists(os.path.realpath(filepath)):
                os.remove(filepath)
                removed += 1
            else:
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                except Exception:
                    os.remove(filepath)
                    removed += 1
    print(f"Cleaned {removed} bad files from {directory}")

clean_dataset(DATA_DIR)

# DATASET LOADING
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = train_generator.num_classes
print(f"Found {train_generator.samples} training images across {num_classes} categories: {list(train_generator.class_indices.keys())}")

# MODEL
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
# base_model.trainable = True
# for layer in base_model.layers[:-70]:   # keep only last ~70 layers trainable
#     layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),                         
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TRAINING 
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print(f"Training complete! Best model saved to: {MODEL_PATH}")

# SAMPLE PREDICTION
class_names = list(train_generator.class_indices.keys())
sample_class = random.choice(class_names)
sample_folder = os.path.join(DATA_DIR, sample_class)
sample_image_name = random.choice(os.listdir(sample_folder))
sample_image_path = os.path.join(sample_folder, sample_image_name)

img = image.load_img(sample_image_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

pred = model.predict(img_batch)
predicted_class = class_names[np.argmax(pred)]

plt.imshow(img)
plt.axis('off')
plt.title(f"Predicted: {predicted_class}  |  Actual: {sample_class}")
plt.show()
