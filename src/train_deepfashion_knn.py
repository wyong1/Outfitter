import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# CONFIG
DATA_DIR = "data/fashion_grouped/train"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# DATASET LOADING
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=False
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

# FEATURE EXTRACTOR
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
model = Model(inputs=base_model.input, outputs=base_model.output)  # 1280-dim features

def extract_features(generator):
    features = model.predict(generator, verbose=1)
    labels = generator.classes
    return features, labels

X_train, y_train = extract_features(train_generator)
X_val, y_val = extract_features(val_generator)

print(f"\nTrain features: {X_train.shape}, Validation features: {X_val.shape}")

# CLASSICAL CLASSIFIER
print("\nTraining classifiers on extracted features...")

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
acc_lr = accuracy_score(y_val, logreg.predict(X_val))
print(f"Logistic Regression Accuracy: {acc_lr:.4f}")

# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
acc_knn = accuracy_score(y_val, knn.predict(X_val))
print(f"KNN Accuracy: {acc_knn:.4f}")

# Support Vector Machine
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
acc_svm = accuracy_score(y_val, svm.predict(X_val))
print(f"SVM Accuracy: {acc_svm:.4f}")

# SAMPLE PREDICTION
print("\nRunning example prediction...")

class_names = list(train_generator.class_indices.keys())
sample_class = random.choice(class_names)
sample_folder = os.path.join(DATA_DIR, sample_class)
sample_image_name = random.choice(os.listdir(sample_folder))
sample_image_path = os.path.join(sample_folder, sample_image_name)

# Load and preprocess image
img = keras_image.load_img(sample_image_path, target_size=IMG_SIZE)
img_array = keras_image.img_to_array(img)
img_array = preprocess_input(img_array)
img_batch = np.expand_dims(img_array, axis=0)

# Extract features and predict
feature = model.predict(img_batch)
predicted_class_idx = int(svm.predict(feature)[0])
predicted_class = list(train_generator.class_indices.keys())[predicted_class_idx]

# Display result
plt.imshow(keras_image.array_to_img(img))
plt.axis('off')
plt.title(f"Predicted: {predicted_class}  |  Actual: {sample_class}")
plt.show()
