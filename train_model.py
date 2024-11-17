import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------------------------------------

image_path = "C:/Data/DJ/azcopydata/midasmultimodalimagedatasetforaibasedskincancer"
images = []

for filename in os.listdir(image_path):
  if filename.endswith(('.jpg', '.jpeg', '.png')):
      img_path = os.path.join(image_path, filename)
      images.append(img_path)

df = pd.read_excel('C:/Data/DJ/azcopydata/midasmultimodalimagedatasetforaibasedskincancer/release_midas.xlsx')
midas_dict = dict(zip(df['midas_file_name'], df['midas_melanoma']))

encoded_label_dict = {key: 1 if value == "yes" else 0 for key, value in midas_dict.items()}

image_folder = "C:/Data/DJ/azcopydata/midasmultimodalimagedatasetforaibasedskincancer"
label_dict = encoded_label_dict

label_dict = {key: "yes" if value == 1 else "no" for key, value in label_dict.items()}

image_paths = [os.path.join(image_folder, image) for image in label_dict.keys()]
labels = list(label_dict.values())

train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# --------------------------------------------------------------------------------------------------------------------
IMG_SIZE = 224

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_paths, 'class': train_labels}),
    x_col="filename",
    y_col="class",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='binary',  # Binary classification: 1 or 0
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': val_paths, 'class': val_labels}),
    x_col="filename",
    y_col="class",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# building model


base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

base_model.trainable = False

#ResNet18
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Unfreezing the last 50 layers of the base ResNet50 model
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# --------------------------------------------------------------------------------------------------------------------

val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")

# model.save('/content/drive/MyDrive/SkinCancerModels/skin_cancer_model_v3.h5')


# --------------------------------------------------------------------------------------------------------------------


# 
# import matplotlib.pyplot as plt

# train_loss = history_finetune.history['loss']
# val_loss = history_finetune.history['val_loss']
# train_acc = history_finetune.history['accuracy']
# val_acc = history_finetune.history['val_accuracy']

# epochs = range(1, len(train_loss) + 1)

# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(epochs, train_loss, label='Training Loss')
# plt.plot(epochs, val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(epochs, train_acc, label='Training Accuracy')
# plt.plot(epochs, val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()