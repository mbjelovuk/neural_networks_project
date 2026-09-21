import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.src.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory
from keras import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import shutil
import os
import imagehash

def get_predictions_and_labels(dataset):
    labels = np.array([])
    pred = np.array([])
    for img, lab in dataset:
        pred_batch = model.predict(img, verbose=0)
        labels = np.concatenate([labels, lab.numpy()])
        pred = np.concatenate([pred, np.argmax(pred_batch, axis=1)])
    return labels, pred

def plot_good_bad_examples(dataset, labels, pred, n_examples=2):
    plt.figure(figsize=(12, 6))
    good_idx = np.where(labels == pred)[0][:n_examples]
    bad_idx = np.where(labels != pred)[0][:n_examples]
    for i, idx in enumerate(good_idx):
        img, lab = list(dataset.unbatch().as_numpy_iterator())[idx]
        plt.subplot(2, n_examples, i+1)
        plt.imshow(img.astype('uint8'))
        plt.title(f"Dobro: {classes[int(lab)]} -> {classes[int(pred[idx])]}")
        plt.axis('off')
    for i, idx in enumerate(bad_idx):
        img, lab = list(dataset.unbatch().as_numpy_iterator())[idx]
        plt.subplot(2, n_examples, i+n_examples+1)
        plt.imshow(img.astype('uint8'))
        plt.title(f"Loše: {classes[int(lab)]} -> {classes[int(pred[idx])]}")
        plt.axis('off')
    plt.show()


main_path = './multi_class_dataset/'

train_dir = os.path.join(main_path, 'Train')
test_dir = os.path.join(main_path, 'Test')

#podela dataseta na trening i test
# if not os.path.exists(train_dir):
#     os.makedirs(train_dir)
# if not os.path.exists(test_dir):
#     os.makedirs(test_dir)
#
# class_folders = [f for f in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, f))]
# train_data = {}
# test_data = {}
#
# for class_folder in class_folders:
#     class_path = os.path.join(main_path, class_folder)
#     images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     print(f"Broj slika u {class_folder}: {len(images)} (putanja: {class_path})")  # Detaljniji ispis
#     if len(images) == 0:
#         print(f"Upozorenje: Klasa {class_folder} je prazna ili nema podržane ekstenzije, preskačem.")
#         continue
#     train_img, test_img = train_test_split(images, test_size=0.2, random_state=123, stratify=[class_folder] * len(images))
#
#     train_data[class_folder] = train_img
#     test_data[class_folder] = test_img
#
#     #podfolderi
#     for img in train_img:
#         dest_dir = os.path.join(train_dir, class_folder)
#         if not os.path.exists(dest_dir):
#             os.makedirs(dest_dir)
#         shutil.move(img, os.path.join(dest_dir, os.path.basename(img)))
#
#     for img in test_img:
#         dest_dir = os.path.join(test_dir, class_folder)
#         if not os.path.exists(dest_dir):
#             os.makedirs(dest_dir)
#         shutil.move(img, os.path.join(dest_dir, os.path.basename(img)))
#
# print(f"Train slike: {sum(len(v) for v in train_data.values())}")
# print(f"Test slike: {sum(len(v) for v in test_data.values())}")

tf.get_logger().setLevel(logging.ERROR)

plt.figure()
klase = ['Glioma', 'Meningioma', 'Pituitary', 'Healthy']
bru = [3325, 3266, 2974, 6704]
plt.bar(klase, bru)
plt.title('Broj uzoraka po klasama')
plt.xlabel('Klase')
plt.ylabel('Broj uzoraka')
plt.show()

img_size = (240, 240)
batch_size = 64

Xtrain = image_dataset_from_directory(
    train_dir,
    subset='training',
    validation_split=0.2,
    image_size=img_size,
    batch_size=batch_size,
    seed=123
)
Xval = image_dataset_from_directory(
    train_dir,
    subset='validation',
    validation_split=0.2,
    image_size=img_size,
    batch_size=batch_size,
    seed=123
)
Xtest = image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    seed=123
)

classes = Xtrain.class_names
print(classes)

N = 4
plt.figure(figsize=(12, 6))
for img, lab in Xtrain.take(1):
    unique_labels = np.unique(lab.numpy())
    for i, label in enumerate(unique_labels):
        idx = np.where(lab.numpy() == label)[0][0]
        plt.subplot(2, 2, i+1)
        plt.imshow(img[idx].numpy().astype('uint8'))
        plt.title(classes[label])
        plt.axis('off')
plt.show()

data_augmentation = Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2)
])

N = 10
plt.figure(figsize=(12, 6))
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img, training=True)
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')
plt.show()

num_classes = len(classes)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(img_size[0], img_size[1], 3)),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1, restore_best_weights=True)

history = model.fit(
    Xtrain,
    epochs=50,
    validation_data=Xval,
    class_weight={0: 16269/3325, 1: 16269/3266, 2: 16269/2974, 3: 16269/6704},
    callbacks=[es],
    verbose=1
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

train_labels, train_pred = get_predictions_and_labels(Xtrain)
val_labels, val_pred = get_predictions_and_labels(Xval)
test_labels, test_pred = get_predictions_and_labels(Xtest)

train_accuracy = 100 * accuracy_score(train_labels, train_pred)
val_accuracy = 100 * accuracy_score(val_labels, val_pred)
test_accuracy = 100 * accuracy_score(test_labels, test_pred)

print(f"Tačnost na trening skupu: {train_accuracy:.2f}%")
print(f"Tačnost na validacionom skupu: {val_accuracy:.2f}%")
print(f"Tačnost na test skupu: {test_accuracy:.2f}%")

print("\nMetrike za validacioni skup:")
print(classification_report(val_labels, val_pred, target_names=classes))

plt.figure()
cm_train = confusion_matrix(train_labels, train_pred, normalize='true')
cm_display_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=classes)
cm_display_train.plot()
plt.title('Confusion Matrix - Training Set', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.show()

plt.figure()
cm_test = confusion_matrix(test_labels, test_pred, normalize='true')
cm_display_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=classes)
cm_display_test.plot()
plt.title('Confusion Matrix - Test Set', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.show()

print("Primeri dobro i loše klasifikovanih slika (validacioni skup):")
plot_good_bad_examples(Xval, val_labels, val_pred)
