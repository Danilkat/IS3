from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as  tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np

# Вводим данные для обучения модели
dollar_q    = np.array([1, 2, 3, 100, 200, 300, 67], dtype=float)
euro_a = np.array([0.82, 1.63, 2.45, 81.75, 163.50, 245.24, 54.77], dtype=float)

for i,c in enumerate(dollar_q):
  print(c, "долларов США =", euro_a[i], "евро")

# Содаем модель
# Используем модель плотной сети (Dense-сеть),
# которая будет состоять из единственного слоя с еднственым нейроном

model = tf.keras.Sequential() #Создаём модель
model.add(tf.keras.layers.Dense(units=1, input_shape=[1])) #Добавляем в модель слой
#Units - количество нейронов
#input_shape - размерность входного параметра (в нашем случае это скаляр)

# Компилируем модель с функцией потерь и оптимизаций

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
# Функция ошибки - среднеквадратичная
# Для функции оптимизации параметр, коэфициент скорости ибучения, равен 0.1
# - это размер шага при корректировке внутренних значений переменных

# Тренируем модель
# используем метод fit, первый аргумент - входные значения, второй арумент - желаемые выходные значения
# epochs - количество итераций цикла обучения
# verbose - контроль уровня логирования

history = model.fit(dollar_q, euro_a, epochs=1000, verbose=0)
print("Завершили тренировку модели")

# Используем модель для предсказаний
predict_set = [228, 322, 500, 1000, 1337]
res_set = model.predict(predict_set);

for i,c in enumerate(predict_set):
  print(c, "долларов США =", *res_set[i], "евро")

# Выводим график обучения
import matplotlib.pyplot as plt
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(history.history['loss'])
plt.show

