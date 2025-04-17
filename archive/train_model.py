import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

#paths
train_dir = '../data/train'
test_dir = '../data/test'

#image Generator
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32)
test_generator = datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32)

#model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train
model.fit(train_generator, validation_data=test_generator, epochs=10)

#save
model.save('../models/apparel_classifier.h5')
