import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model 
import subprocess


from tkinter import messagebox

from PIL import Image 

longitud, altura = 100,100
modelo='./modelo/modelo.h5'
pesos='./modelo/pesos.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)
mensaje1=''
def predict(file):
	x=load_img(file, target_size=(longitud,altura))
	x=img_to_array(x)
	x=np.expand_dims(x,axis=0)
	arreglo=cnn.predict(x)#[1,0,0]
	resultado=arreglo[0]
	respuesta=np.argmax(resultado) 
	if respuesta==0:
		mensaje1 = 'Fase 1 de maduracion\n'
		mensaje1 +='El fruto se encuentra duro\n'
		mensaje1 +='El aguacate tarda en llegar a la siguiente etapa de madurez aproximadamente una semana'
		mensaje1 +='\nPorcentaje aproximado de maduracion de 0 a 20%'
		messagebox.showinfo(message=mensaje1, title="Resultado")
	if respuesta==1:
		mensaje1='Fase 2 de maduracion\n'
		mensaje1+='El aguacate tarda en llegar a la siguiente etapa de madurez aproximadamente una semana'
		mensaje1+='\nPorcentaje aproximado de maduracion de 20 a 40%'
		messagebox.showinfo(message=mensaje1, title="Resultado")
	if respuesta==2:
		mensaje1='Fase 3 de maduracion\n'
		mensaje1+='El aguacate tarda en llegar a la siguiente etapa de madurez aproximadamente 2 semanas'
		mensaje1+='\nNo se recomienda esperar mas tiempo para cosechar, ya que el producto puede sufrir danios en su manejo'
		mensaje1+='\nPorcentaje aproximado de maduracion de 40 a 60%'
		messagebox.showinfo(message=mensaje1, title="Resultado")
	if respuesta==3:
		mensaje1='Fase 4 de maduracion\n'
		mensaje1+='El aguacate se encuentra aproximadamente a de 2 a 1 semana para llegar a su madurez.'
		mensaje1+='\nPorcentaje aproximado de maduracion de 60  80%'
		messagebox.showinfo(message=mensaje1, title="Resultado")
	if respuesta==4:
		mensaje1='Fase 5 de maduracion\n'
		mensaje1+='Caducidad: En al rededor de 2 a 5 dias el aguacate dejara de ser apto para su consumo'
		mensaje1+='\nPorcentaje aproximado de maduracion de 80 a 100%'
		messagebox.showinfo(message=mensaje1, title="Resultado")
	if respuesta==6:
		mensaje1='Mal Estado\n'
		mensaje1='Mala calidad'
		messagebox.showinfo(message=mensaje1, title="Resultado")
	if respuesta==5:
		mensaje1='No es un aguacate'
		messagebox.showinfo(message=mensaje1, title="Resultado")
	mensaje=mensaje1
	return respuesta

print('Ingrese uno de los siguientes archivos')
subprocess.call(['ls'])
ruta = raw_input()
predict(ruta)

image = Image.open(ruta) 
image = image.resize((500, 500), Image.ANTIALIAS)
image.show() 

