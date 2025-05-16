import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, RandomRotation, RandomFlip, RandomContrast, Rescaling, Input
import matplotlib.pyplot as plt
import logging

# Настройка логирования
tf.get_logger().setLevel(logging.ERROR)
keras.utils.set_random_seed(17)

# Параметры
EPOCHS = 128
BATCH_SIZE = 16
DROPOUT_VAL = 0.3

# Загрузка данных
train = keras.utils.image_dataset_from_directory(
    directory='lab 10/dataset/training',
    labels='inferred',
    label_mode='categorical',
    image_size=(512, 512))

validate = keras.utils.image_dataset_from_directory(
    directory='lab 10/dataset/validation',
    labels='inferred',
    label_mode='categorical',
    image_size=(512, 512))

test = keras.utils.image_dataset_from_directory(
    directory='lab 10/dataset/test',
    labels='inferred',
    label_mode='categorical',
    image_size=(512, 512))

# Аугментация данных
data_augmentation = keras.Sequential(
    [
        RandomRotation(factor=0.1),
        RandomFlip("horizontal"),
        RandomContrast(factor=0.1),
    ]
)

regularizer = keras.regularizers.L2(l2=0.01)
MODEL_DIRECTION = 'lab 10/model.keras'

# Загрузка предобученной модели
base_model = tf.keras.models.load_model(MODEL_DIRECTION)
base_model.trainable = False

# Создание модели
inputs = keras.Input(shape=(512, 512, 3))
x = data_augmentation(inputs)
x = MaxPooling2D((4, 4), strides=8)(x)
x = base_model(x, training=False)
output = Dense(2, activation="sigmoid", kernel_initializer="glorot_uniform", bias_initializer="zeros")(x)

model = keras.Model(inputs, output)

# Компиляция модели
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Вывод саммари модели в консоль
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

# Вывод точности и потерь для первых 4 эпох
print("\nИстория обучения (первые 4 эпохи):")
for epoch in range(min(4, EPOCHS)):
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {history.history['accuracy'][epoch]:.4f} - Train Loss: {history.history['loss'][epoch]:.4f} - Val Acc: {history.history['val_accuracy'][epoch]:.4f} - Val Loss: {history.history['val_loss'][epoch]:.4f}")

# Визуализация точности
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'])
plt.savefig('accuracy_plot.png')
plt.close()

# Визуализация потерь
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'validation'])
plt.savefig('loss_plot.png')
plt.close()

# Оценка модели
print("\nОценка модели:")
val_metrics = model.evaluate(validate, verbose=2)
print(f"Валидация: Точность = {val_metrics[1]:.4f}, Потери = {val_metrics[0]:.4f}")
test_metrics = model.evaluate(test, verbose=2)
print(f"Тест: Точность = {test_metrics[1]:.4f}, Потери = {test_metrics[0]:.4f}")

# Визуализация предсказаний
img = list(test)[0]
x_out = model.predict(img[0], verbose=2)

fig, axes = plt.subplots(4, 5, figsize=(8, 8))
for i in range(20):
    predicted_class = 'Cat' if x_out[i].argmax() == 0 else 'Dog'
    true_class = 'Cat' if img[1][i].numpy().argmax() == 0 else 'Dog'
    sign = '+' if x_out[i].argmax() == img[1][i].numpy().argmax() else '-'
    row = i // 5
    col = i % 5
    ax = axes[row, col]
    ax.imshow(img[0][i].numpy().astype("uint8"))
    ax.axis("off")
    ax.set_title(f"{predicted_class} ({sign})", fontsize=12)
plt.tight_layout()
plt.savefig('predictions_plot.png')
plt.close()