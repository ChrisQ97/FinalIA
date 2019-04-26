import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator #ayuda a preprocesar las imagenes para entrenar
from tensorflow.python.keras import optimizers #optimizador
from tensorflow.python.keras.models import Sequential #libreria que permite hacer redes secuenciales, cada una en orden
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation #
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D #capas para hacer las convoluciones y el pooling
from tensorflow.python.keras import backend as K #revisa si queras esta en segundo proceso y lo mata

K.clear_session() #matamos el keras

data_entrenamiento = './datos/entrenamiento'
data_validacion = './datos/validacion'

#parametros
epocas = 30 #cantidad de iteraciones sobre todo el set de datos
altura, longitud = 100, 100
batch_size=32 #numero de imagenes que se van a mandar a procesar en cada paso
pasos = 1000 #numero de pasos que se van a realizar en cada epoca
pasos_de_validacion = 200 #para ver que si funciona el algoritmo al final se corren 200 paso con los datos de validacion para ver que tan bien aprende
filtrosConv1=32 #numero de filtros que se aplcian en cada convolucion
filtrosConv2=64 #despues del primer filtro tendra 64 de profundidad
tamanio_filtro1 = (3,3) #para la priemra convolucion el filtro tendra 3,3
tamanio_filtro2 = (2,2) #para la segunda 2,2
tamanio_pool=(2,2) #tama;o de filtro de maxpooling
clases=7 #cantidad de etapas
lr=0.0005 #tama;o de ajustes.

#preprocesamiento de imagenes
#despues se pasan a la red neuronal
#transformacion de imagenes

#generador el de la linea 3
entrenamiento_datagen = ImageDataGenerator(
	rescale=1./255, #reescalado de colores de 0 a 1
	shear_range=0.3, #inclina la imagen
	zoom_range=0.3, #a algunas les hace zoom para que no siempre este completo
	horizontal_flip=True #toma la imagen y la invierte
)

#con validacion
validacion_datagen = ImageDataGenerator(
	rescale=1./255
)#solo nos interesa reescalarlas
#las siguientes variables tendran las imagenes
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory( 
	data_entrenamiento,
	target_size=(altura,longitud),
	batch_size =batch_size,
	class_mode='categorical'
)
#entra en todas las carpetas y las procesa

imagen_validacion=validacion_datagen.flow_from_directory(
	data_validacion,
	target_size=(altura,longitud),
	batch_size=batch_size,
	class_mode='categorical'
)
#creando red CNN
cnn=Sequential()
cnn.add(Convolution2D(filtrosConv1,tamanio_filtro1,padding='same',input_shape=(altura,longitud,3),activation='relu'))
#						nofiltros, tamani filtros, lo que se hace en las esquinas, dimensiones, funcion de activacion
cnn.add(MaxPooling2D(pool_size=tamanio_pool)) 
cnn.add(Convolution2D(filtrosConv2,tamanio_filtro2,padding='same',activation='relu'))#input shape solo en la primer capa
cnn.add(MaxPooling2D(pool_size=tamanio_pool))

#empezar clasificacion
cnn.add(Flatten())#esa imagen que ahroa es profunda se hace plana en una solo dimension con toda la informacion de la red neuronal
#mandamos a una capa normal
cnn.add(Dense(256,activation='relu'))


cnn.add(Dropout(0.5))#a la capa densa durante el entrenamiento se apaga el 50% de las neuronas para evitar desajustar
cnn.add(Dense(clases,activation='softmax'))

cnn.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=lr),metrics=['accuracy'])
#vea que tan bien o que tan mal va
#la metrica para ver que tan bien aprende es accuracy % de que tan bien aprende

cnn.fit(imagen_entrenamiento,steps_per_epoch=pasos,epochs=epocas,validation_data=imagen_validacion,validation_steps=pasos_de_validacion)

dir='./modelo/'
if not os.path.exists(dir):
	os.mkdir(dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')
print('***************************************')
#print(ImageDataGenerator.class_indices)