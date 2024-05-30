import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from PIL import Image
from google.colab import files

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Предобработка данных
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
#print(x_test)
#print(y_test)
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=6)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

# Определение точности модели
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Точность модели на тестовых данных: %.2f%%' % (test_accuracy * 100))


uploaded = files.upload()
image_path = list(uploaded.keys())[0]
image = Image.open(image_path).convert('L')
image = image.resize((28, 28))
image = np.array(image)
image = image.reshape((1, 28, 28, 1))
image = image.astype('float32') / 255

# Предсказание модели на своем изображении
prediction = model.predict(image)
predicted_label = np.argmax(prediction[0])

print('Предсказанная цифра: %d' % predicted_label)