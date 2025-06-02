import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, RandomRotation, RandomFlip, RandomContrast, Rescaling, Input, RandomZoom
import matplotlib.pyplot as plt
import logging
import os
import numpy as np

# Настройка логирования и случайного seed
tf.get_logger().setLevel(logging.ERROR)
keras.utils.set_random_seed(228)

# Параметры обучения
EPOCHS = 50
BATCH_SIZE = 16
DROPOUT_VAL = 0.3
IMG_SIZE = (128, 128)

# Загрузка данных
train = keras.utils.image_dataset_from_directory(
    directory='lab 12/dataset/training',
    labels='inferred',
    label_mode='categorical',
    class_names=['botanika', 'sel\'hoz', 'nichego'],
    image_size=IMG_SIZE
)

validate = keras.utils.image_dataset_from_directory(
    directory='lab 12/dataset/validation',
    labels='inferred',
    label_mode='categorical',
    class_names=['botanika', 'sel\'hoz', 'nichego'],
    image_size=IMG_SIZE
)

test = keras.utils.image_dataset_from_directory(
    directory='lab 12/dataset/test',
    labels='inferred',
    label_mode='categorical',
    class_names=['botanika', 'sel\'hoz', 'nichego'],
    image_size=IMG_SIZE
)

# Аугментация данных
data_augmentation = keras.Sequential([
    RandomRotation(factor=0.1),
    RandomFlip("horizontal"),
    RandomContrast(factor=0.1),
    RandomZoom(height_factor=0.1, width_factor=0.1)
])

# Регуляризация L2
regularizer = keras.regularizers.L2(l2=0.01)

# Создание модели
model = Sequential()
model.add(Input(shape=(128, 128, 3)))
model.add(data_augmentation)
model.add(Rescaling(scale=1./255))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=regularizer))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=regularizer))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=regularizer))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'))
model.add(Dropout(DROPOUT_VAL))
model.add(Dense(3, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros'))

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Обучение модели
history = model.fit(
    train,
    validation_data=validate,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
    shuffle=True
)

# Построение графиков точности и потерь
plt.figure(figsize=(12, 4))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

# График потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.show()

# Оценка модели на проверочных и тестовых данных
print("Evaluation on validation data:")
model.evaluate(validate)

print("Evaluation on test data:")
model.evaluate(test)

# Визуализация результатов классификации
batch = list(test)[0]
x_out = model.predict(batch[0], verbose=2)

# Отображение результатов
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
class_names = ['botanika', 'sel\'hoz', 'nichego']
for i in range(20):
    prediction = x_out[i].argmax()
    actual = batch[1][i].numpy().argmax()
    correct = prediction == actual
    
    row = i // 5
    col = i % 5
    ax = axes[row, col]
    
    ax.imshow(batch[0][i].numpy().astype("uint8"))
    ax.axis("off")
    
    title = class_names[prediction]
    sign = "+" if correct else "-"
    confidence = x_out[i][prediction]
    ax.set_title(f"{title} ({sign}, {confidence:.2f})", fontsize=12)

plt.tight_layout()
plt.show()

new_images_path = 'lab 11/dataset/new_images'

new_images = []
for filename in os.listdir(new_images_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(new_images_path, filename)
        img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0
        new_images.append(img_array)

# Предсказание для каждого изображения
for i, img_array in enumerate(new_images):
    prediction = model.predict(img_array, verbose=0)
    predicted_class = class_names[prediction[0].argmax()]
    confidence = prediction[0][prediction[0].argmax()]

    # Отображение изображения и предсказания
    img = keras.utils.load_img(os.path.join(new_images_path, os.listdir(new_images_path)[i]), target_size=IMG_SIZE)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Предсказание: {predicted_class} (Уверенность: {confidence:.2f})")
    plt.show()