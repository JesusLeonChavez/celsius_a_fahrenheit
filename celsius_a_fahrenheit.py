import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Importando datos
temperature_df = pd.read_csv("celsius_a_fahrenheit.csv")

#visualizacion
sns.scatterplot(temperature_df['Celsius'], temperature_df['Fahrenheit'])

#cargando set de datos
x_train = temperature_df['Celsius']
y_train = temperature_df['Fahrenheit']

#creando el modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 1, input_shape=[1]))

#model.summary()

#Compilado
model.compile(optimizer = tf.keras.optimizers.Adam(1), loss='mean_squared_error')

#entrenamiento del modelo
epochs_hist = model.fit(x_train, y_train, epochs = 100)

#evaluando el modelo
epochs_hist.history.keys()

#Grafico
plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de perdida del modelo entrenamiento del modelo')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend('Training Loss')

#Predicciones
Temp_C = 0
Temp_F = model.predict([Temp_C])
print(Temp_F)


Temp_F = 9/5 * Temp_C + 32
print(Temp_F)


