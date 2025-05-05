import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras import regularizers
import numpy as np
import logging

tf.get_logger().setLevel(logging.ERROR)  # подавляет информационные и отладочные сообщения

EPOCHS = 500
BATCH_SIZE = 16
DROPOUT_VAL = 0.2

bostonhousing = keras.datasets.boston_housing
regularizer = regularizers.L2(0.1)
(raw_x_train, y_train), (raw_x_test, y_test) = bostonhousing.load_data()
x_mean = np.mean(raw_x_train, axis=0)  # вычисляет среднее значение каждого признака, axis=0 - среднее берется по строкам
x_stddev = np.std(raw_x_train, axis=0)  # насколько сильно значение признака отклоняется от среднего значения
x_train = (raw_x_train - x_mean) / x_stddev  # нормализация
x_test = (raw_x_test - x_mean) / x_stddev

# Листинг 1 - Исходная модель
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=[13]))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='MSE', optimizer='adam', metrics=['MAE'])
model.summary()
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)
predictions = model.predict(x_test)
for i in range(0, 4):
    print('Prediction: ', predictions[i], ', true value: ', y_test[i], ', MSE: ', history.history['loss'][i])

# Листинг 2 - Модель с одним слоем, одним нейроном и линейной функцией
layermodel = Sequential()
layermodel.add(Dense(1, activation='linear', input_shape=[13]))
layermodel.compile(loss='MSE', optimizer='adam', metrics=['MAE'])
layermodel.summary()
history_layer = layermodel.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)
predictions_layer = layermodel.predict(x_test)
for i in range(0, 4):
    print('Prediction: ', predictions_layer[i], ', true value: ', y_test[i], ', MSE: ', history_layer.history['loss'][i])

# Листинг 3 - Модель с модификацией “a)” (L2-регуляризация)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=[13], kernel_regularizer=regularizer))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizer))
model.add(Dense(1, activation='linear', kernel_regularizer=regularizer))
model.compile(loss='MSE', optimizer='adam', metrics=['MAE'])
model.summary()
history_a = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)
predictions_a = model.predict(x_test)
for i in range(0, 4):
    print('Prediction: ', predictions_a[i], ', true value: ', y_test[i], ', MSE: ', history_a.history['loss'][i])

# Листинг 4 - Модель с модификацией “b)” (Dropout 0.2)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=[13]))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='MSE', optimizer='adam', metrics=['MAE'])
model.summary()
history_b = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)
predictions_b = model.predict(x_test)
for i in range(0, 4):
    print('Prediction: ', predictions_b[i], ', true value: ', y_test[i], ', MSE: ', history_b.history['loss'][i])

# Листинг 5 - Модель с модификацией “c)” (дополнительный слой, 128 нейронов)
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=[13]))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='MSE', optimizer='adam', metrics=['MAE'])
model.summary()
history_c = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)
predictions_c = model.predict(x_test)
for i in range(0, 4):
    print('Prediction: ', predictions_c[i], ', true value: ', y_test[i], ', MSE: ', history_c.history['loss'][i])

# Листинг 6 - Модель с модификацией “d)” (Dropout 0.3)
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=[13]))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))
model.compile(loss='MSE', optimizer='adam', metrics=['MAE'])
model.summary()
history_d = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)
predictions_d = model.predict(x_test)
for i in range(0, 4):
    print('Prediction: ', predictions_d[i], ', true value: ', y_test[i], ', MSE: ', history_d.history['loss'][i])