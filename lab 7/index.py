import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import logging

# Настройки логгирования и генератора случайных чисел
tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(7)

# Гиперпараметры
EPOCHS = 20
BATCH_SIZE = 32

# Загрузка данных MNIST
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Стандартизация данных
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

# One-hot кодирование меток
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Инициализатор весов
initializer = keras.initializers.he_normal()

# Создание модели с 2 скрытыми слоями
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),

    # Скрытый слой 1
    keras.layers.Dense(64, kernel_initializer=initializer, bias_initializer='zeros'),
    keras.layers.Activation('sigmoid'),
    keras.layers.BatchNormalization(),

    # Скрытый слой 2
    keras.layers.Dense(32, kernel_initializer=initializer, bias_initializer='zeros'),
    keras.layers.Activation('sigmoid'),
    keras.layers.BatchNormalization(),

    # Выходной слой
    keras.layers.Dense(10, kernel_initializer=initializer, bias_initializer='zeros'),
    keras.layers.Activation('sigmoid')
])

# Вывод структуры модели
model.summary()

# Оптимизатор и компиляция модели
opt = keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss='MSE', optimizer=opt, metrics=['accuracy'])

# Обучение модели
history = model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
    shuffle=True
)

# Вывод финальной точности
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"Финальная точность на обучающем наборе: {final_train_accuracy:.4f}")
print(f"Финальная точность на тестовом наборе: {final_val_accuracy:.4f}")

# Оценка на тестовом наборе
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"Точность на тестовом наборе: {test_accuracy:.4f}")

# Построение графиков точности и потерь
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Точность')
plt.xlabel('Эпохи')
plt.ylabel('Accuracy')
plt.legend(['Обучение', 'Валидация'])
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Потери')
plt.xlabel('Эпохи')
plt.ylabel('Loss')
plt.legend(['Обучение', 'Валидация'])
plt.show()

# Тестирование модели на 36 примерах
x_out = model.predict(test_images[:36], verbose=2)

fig, axes = plt.subplots(6, 6, figsize=(8, 8))
for i in range(36):
    if x_out[i].argmax() == test_labels[i].argmax():
        sign, color = "+", "green"
    else:
        sign, color = "-", "red"

    row = i // 6
    col = i % 6
    number = test_images[i].reshape(28, 28)
    ax = axes[row, col]
    ax.imshow(number, cmap=plt.cm.binary)
    ax.axis("off")
    ax.set_title(f"{x_out[i].argmax()} ({sign})", fontsize=12)
    ax.add_patch(plt.Rectangle((0, 0), 27, 27, fill=False, edgecolor=color, linewidth=2))
    ax.title.set_color(color)

plt.tight_layout()
plt.show()