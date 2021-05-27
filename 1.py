# from idlelib import history
# import _tkinter
import inline
import matplotlib
import pylab as pl
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# загрузим набор данных
# x_train - тренировочный набор, y_train - правильные ответы
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# создадим список с названием классов
classes = ['футболка', 'брюки','свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботанки']

'''
# просмотрим примеры изображений
plt.figure(figsize=(10,10))
for i in range(100, 150):
    plt.subplot(5,10,i-100+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(classes[y_train[i]])
'''

# преобразуем размерности данных в ноборе (из двумерного в плоский)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# нормализуем данные, векторизованные операции применяются к каждому элементу массива отдельно
x_train = x_train/255
x_test = x_test/255


# подготовим правильный ответ
n = 3
print(y_train[n])

# преобразуем метки в формат one hot encoding
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# правильный ответ в формате one hot encoding
print(y_train[n])


# создаем нейронную сеть
# создадим объект последовательной нейронной сети
model = Sequential

# Поместим необходимые слои в модель
model.add(Dense(800, input_dim=784, activation='relu'))

# Выходной полносвязанный слой, 10 нейронов (по колличеству рукописных цифр)
model.add(Dense(10, activation='softmax'))

# скомпилируем сеть: loss - категория ошибки, optimizer - алгоритм обучения, metrics - метрика качества обучения сети(
# доля правильныз ответов)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# обучим нейронную сеть на минивыборке:batch_size-размер, epochs-колличество эпох, verbose-напечате прогресс обучения
# сети, validation_split-создает проверочный набор данных)
model.fit(x_train, y_train,
          batch_size=200,
          epochs=50,
          validation_split=0.2,
          verbose=1)
'''
# создадим график качества обучения
plt.plot(history.history['accuracy'],
         label='Доля правильных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля правильных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля правильных ответов')
plt.legend()
pl.show()
'''

# сохраним созданную нейронную сеть для дальнейшего использования
model.save('fashion_mnist_dense.hs')


# проверим качество работы на наборе данных для тестирования
scores = model.evaluate(x_test, y_test, verbose=1)
print('Доля правильных ответов на тестовых анных в процентах:', round(scores[1] * 100, 4))

'''
# запустим распознование набора на котором было обучение
predictions = model.predict(x_train)

# просмотрим пример изображения
# меняйте значение n чтобы просмотреть результаты распознования других изображений
n = 0
plt.imshow(x_train[n].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

# определим номер классаизображения, который предлагает сеть
np.argmax(predictions[n])

# напечатаем название класса
classes[np.argmax(predictions[n])]

# печатаем номер класса правильного ответа
np.argmax(y_train)

# печатаем название класса правильного ответа
classes[np.argmax(y_train[n])]
'''