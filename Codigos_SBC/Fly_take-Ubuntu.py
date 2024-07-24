# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:17:09 2024

@author: felip
"""

import cv2
import numpy as np
import math
import serial
from ultralytics import YOLO
import random
import time
import threading
import os
from datetime import datetime
import queue
import string

# Variable global para controlar el bucle
exit_flag = False


def generate_random_string(length):
    """
    Genera una cadena aleatoria de longitud dada.
    
    Parámetros:
    - length: longitud de la cadena.
    """
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for _ in range(length))

def check_input():
    global exit_flag
    # Esperar 60 segundos (1 minuto)
    time.sleep(25)
    exit_flag = True

def grabar_video(cap, stop_event, filename="output.mp4"):
    """
    Función para grabar video en un archivo en un hilo separado.
    
    Parámetros:
    - cap: objeto VideoCapture de OpenCV.
    - stop_event: threading.Event para detener la grabación.
    - filename: nombre del archivo de salida.
    """
    # Directorio donde se guardará el video
    output_dir = "/home/khadas/Desktop/Datos"

    # Crear el directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generar el nombre de archivo aleatorio
    random_suffix = generate_random_string(6)
    filename = f"output_{random_suffix}.mp4"

    # Ruta completa del archivo de salida
    filepath = os.path.join(output_dir, filename)

    # Obtener la resolución de la cámara
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Definir el codec y crear el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 20.0, (width, height))

    while not stop_event.is_set():
        ret, frame = cap.read()
        frame_rotado = cv2.rotate(frame, cv2.ROTATE_180)
        
        if not ret:
            print("Error: No se puede recibir frames. Terminando...")
            break

        # Dibujar un círculo en el frame
        cv2.circle(frame_rotado, (320, 240), 5, (0, 0, 255), -1)

        out.write(frame_rotado)
        # Agregar el frame recibido del hilo principal
        try:
            captured_frame = frame_queue.get_nowait()
            if captured_frame is not None:
                for i in range(20):
                    out.write(captured_frame)
        except queue.Empty:
            pass

    out.release()
    print(f"Video guardado en {filepath}")


def to_angulos(lista_duplas):
      """
      Función que transforma una lista de duplas dividiendo el primer elemento
      por 1.8, retornando una lista con los primeros elementos transformados.
    
      Parámetros:
        lista_duplas: Lista de duplas.
    
      Retorno:
        Lista con los primeros elementos de las duplas transformados.
      """
    
      lista_transformada = []
      for dupla in lista_duplas:
        x_transformado = dupla[0] / 1.8
        lista_transformada.append(x_transformado)
    
      return lista_transformada

def dibujar_radar(frame, angulos,ddd):
     """
     Función para dibujar el radar con una lista de ángulos y una lista de distancias aleatorias, asignando colores aleatorios a los puntos.
    
     Parámetros:
      frame: Imagen original.
      angulos: Lista de ángulos en grados.
    
     Retorno:
      Imagen con el radar dibujado y los puntos correspondientes a los ángulos y la distancia variable con colores aleatorios.
     """
     frame_negro = np.zeros((480, 640, 3), dtype=np.uint8)
     # Crear un canvas negro del mismo tamaño que el frame
     canvas = np.zeros_like(frame_negro)
    
     # Dibujar el círculo exterior del radar
     cv2.circle(canvas, (320, 240), 200, (255, 255, 255), 2)
    
     # Dibujar las líneas radiales del radar
     for i in range(0, 360, 10):
      rad = math.radians(i)
      x1 = int(320 + 100 * math.cos(rad))
      y1 = int(240 + 100 * math.sin(rad))
      x2 = int(320 + 125 * math.cos(rad))
      y2 = int(240 + 125 * math.sin(rad))
      cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)
    
     # Dibujar el eje horizontal y vertical del radar
     cv2.line(canvas, (120, 240), (520, 240), (255, 255, 255), 2)
     cv2.line(canvas, (320, 40), (320, 440), (255, 255, 255), 2)
    
     # Dibujar el punto central del radar
     cv2.circle(canvas, (320, 240), 3, (255, 255, 255), -1)
    
     # Convertir la lista de ángulos a radianes
     radianes = [math.radians(angulo + 270) for angulo in angulos]
    
     # Generar una lista de distancias aleatorias
     random.seed(2)
     distancia2 = ddd
     print("Distancia a los objetos: ",distancia2)
     print("Angulo a los objetos: ",angulos)
     # Calcular las coordenadas cartesianas de los puntos
     puntos = [(int(320 + distancia * math.cos(angulo)), int(240 + distancia * math.sin(angulo))) for distancia, angulo in zip(distancia2, radianes)]
    
     # Dibujar los puntos en el radar con colores aleatorios
     for punto in puntos:
       # Generar colores aleatorios para los puntos
      color1= int(random.randint(0, 255))
      color2=int(random.randint(0, 255))
      color3=int(random.randint(0, 255))
      cv2.circle(canvas, punto, 5,(color1,color2,color3), -1)
    
     # # Escribir el ángulo y la distancia como texto
     # for i, angulo in enumerate(angulos):
     #  texto = f"Angulo {i+1}: {angulo}° Distancia {i+1}: {i+1}m"
     #  text_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
     #  cv2.putText(canvas, texto, (punto[0] - text_size[0] // 2, punto[1] - 10 - (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1)
    
     return canvas




def draw_circle(event, x, y, flags, param):
    global puntos, mover
    if event == cv2.EVENT_LBUTTONDBLCLK:
        
        puntos.append((x, y))
        mover=True




def to_pwm(duplas):
    pwms = []
    for x,y in duplas:
        x_relative = x - 320
        y_relative = y - 220
        x_angle = angle_per_pixel_horizontal * x_relative
        y_angle = angle_per_pixel_vertical * y_relative
        xpwm = int(x_angle * 1.75)
        ypwm = int(y_angle * 1.9)
        pwms.append((xpwm, ypwm))
    return pwms

def pwms_C(lista2):
    lista_modificada = []  # Lista vacía para almacenar las diferencias
    lista_modificada.append(lista2[0])  # Agregar el primer elemento de la lista original
    
    for i in range(1, len(lista2)):
        # Calcular la diferencia entre la dupla actual y la anterior
        diferencia = (lista2[i][0] - lista2[i - 1][0], lista2[i][1] - lista2[i - 1][1])
        # Agregar la diferencia a la lista modificada
        lista_modificada.append(diferencia)
    
    return lista_modificada

def enviar_datos_por_serial(duplas):
    dd=[]
    global frame_queue
    for punto in duplas:
        xpwm = punto[0]
        ypwm = punto[1]
        xpwm=abs(xpwm)
        ypwm=abs(ypwm)

        
        # Verificar si xpwm es positivo o negativo
        if punto[0] > 0:
            print("d")
            ser.write(b'd')
            received_data = ser.readline().decode('utf-8')
            ser.write(str(xpwm).encode() + b"\n")
            received_data = ser.readline().decode('utf-8')
        elif punto[0] < 0:
            print("i")
            ser.write(b'i')
            received_data = ser.readline().decode('utf-8')
            ser.write(str(xpwm).encode() + b"\n")
            received_data = ser.readline().decode('utf-8')

        time.sleep(0.2)
        # Verificar si ypwm es positivo o negativo
        if punto[1] > 0:
            print("a")
            ser.write(b'a')
            received_data = ser.readline().decode('utf-8')
            ser.write(str(ypwm).encode() + b"\n")
            received_data = ser.readline().decode('utf-8')
        elif punto[1] < 0:
            print("u")
            ser.write(b'u')
            received_data = ser.readline().decode('utf-8')
            ser.write(str(ypwm).encode() + b"\n")
            received_data = ser.readline().decode('utf-8')
        time.sleep(1)
        ser.write(b'l')
        
        try:
            received_data = ser.readline().decode('utf-8').strip()
            if received_data!="p" or received_data!="y":
                received_data = float(received_data)
                dist = received_data / 100.0
                # Formatear el texto para incluir la medida en metros
                texto = f'{dist:.2f} m'
                print(f'Dato recibido como float: {received_data}')
                
                
                dd.append(dist)
            else:
                print('No se recibió ningún dato')
                received_data = ser.readline().decode('utf-8')
                dist=0
                texto = f'{dist:.2f} m'
        except ValueError:
            print('No se pudo convertir a float. Se recibió:', received_data)
            
        print(f'Datos recibidos: {received_data}')
        ret, frame = cap.read()
        frame_rotado = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.circle(frame_rotado, (320,240), 5, (0, 0, 255), -1)  
        text_position = (320, 240 - 30)  # Ajusta según sea necesario
        cv2.putText(frame_rotado, texto, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (10, 120, 255), 3)
        frame_queue.put(frame_rotado)
        #out.write(frame)
        
        #ruta_archivo = guardar_frame_con_fecha(frame_rotado,3)
        #time.sleep(0.5)
        # while True:
        #     ret, frame = cap.read()
        #     cv2.circle(frame, (320,240), 5, (0, 0, 255), -1)    
        #     #cv2.imshow('Webcam2', frame)
            
    return dd
    

def guardar_frame_con_fecha(frame,nom):
    """
    Función para guardar un frame de OpenCV como una imagen con la fecha actual como parte del nombre del archivo.

    Parámetros:
        frame: El frame de OpenCV que se desea guardar como imagen.
        ruta_carpeta: La ruta de la carpeta donde se guardará la imagen.

    Retorno:
        ruta_completa: La ruta completa del archivo guardado.
    """
    
    ruta_carpeta = "/home/khadas/Desktop/Datos"
    # Obtener la fecha y hora actual
    fecha_actual = datetime.now().strftime("%m%d_%H%M%S")

    if nom==0:
        
        # Combina la fecha actual con el nombre del archivo
        nombre_archivo = f"Radar_{fecha_actual}.jpg"
    elif nom==1:
        nombre_archivo = f"Foto_{fecha_actual}.jpg"
    else:
        nombre_archivo = f"T{nom}_{fecha_actual}.jpg"

    # Combina la ruta de la carpeta con el nombre del archivo
    ruta_completa = f"{ruta_carpeta}/{nombre_archivo}"

    # Guarda el frame como una imagen
    cv2.imwrite(ruta_completa, frame)

    print(f"Imagen guardada en: {ruta_completa}")
    
    return ruta_completa
    
image_width = 640
image_height = 480
IWH=image_width/2
IHH=image_height/2
dFoV = 55  # Campo de visión diagonal en grados

# Calcular campos de visión horizontal y vertical
FOV_horizontal = 2 * math.atan((image_width / 2) / (image_width / (2 * math.tan(math.radians(dFoV / 2))))) * (360 / (2 * math.pi))
FOV_vertical = 2 * math.atan((image_height / 2) / (image_width / (2 * math.tan(math.radians(dFoV / 2))))) * (360 / (2 * math.pi))
angle_per_pixel_horizontal = FOV_horizontal / image_width
angle_per_pixel_vertical = FOV_vertical / image_height



puntos = []
mover=False

port = '/dev/ttyACM0'
baudrate = 115200
ser = serial.Serial(port, baudrate, timeout=1)



cap = cv2.VideoCapture(0)
ret, frame = cap.read()



model = YOLO("best.pt")
#model = YOLO("yolov8s.pt")
count=0
angulos=[0,0]

# radar = dibujar_radar(frame, 0, 0)
# cv2.imshow('Radar', radar)
    # Evento para detener la grabación
stop_event = threading.Event()

    # Crear y arrancar el hilo de grabación de video
grabar_thread = threading.Thread(target=grabar_video, args=(cap, stop_event))
grabar_thread.start()

input_thread = threading.Thread(target=check_input)
input_thread.start()

frame_queue = queue.Queue()

try:
    
    while not exit_flag:
        time.sleep(3)
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        copia=frame
        results = model(frame)
        data_list = []  # Lista para almacenar los datos
        max_area = float("-inf")  # Inicializar el área máxima
        
        if results is not None:
            min_distance = float('inf')
            nearest_box_center = None
            # Iterar sobre los resultados de la inferencia
            for i, r in enumerate(results):
                boxes = r.boxes
            
                for j, box in enumerate(boxes):
                    # Coordenadas del bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convertir a valores enteros
                    
                    # Calcular el centro del bounding box
                    box_center_x = (x1 + x2) // 2
                    box_center_y = (y1 + y2) // 2
                    
                    # Calcular la distancia entre el centro del bounding box y el punto (320, 240)
                    distance = ((box_center_x - 320) ** 2 + (box_center_y - 240) ** 2) ** 0.5
                    
                    # Actualizar el punto más cercano si la distancia es menor
                    if distance < min_distance:
                        min_distance = distance
                        nearest_box_center = (box_center_x, box_center_y)
                    
                    # Calcular el área del bounding box
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Actualizar el área máxima
                    if area > max_area:
                        max_area = area
                    
                    # Guardar los datos en data_list
                    data_list.append((j, (x1 + x2) // 2, (y1 + y2) // 2, x1, y1, x2, y2, area))
            
            # Iterar sobre los datos en data_list para calcular los puntajes
            max_score = float("-inf")
            max_score_data = None
            data_list_score = [] 
            
            for j, centro_x, centro_y, x1, y1, x2, y2, area in data_list:
                # Normalizar y2
                normalized_y2 = y2 / 480
                
                # Normalizar el área usando el área máxima
                normalized_area = area / max_area
                
                # Calcular el puntaje combinado
                score = (normalized_y2 * 0.7) + (normalized_area * 0.3)
                
                # Actualizar el máximo puntaje y los datos asociados
                if score > max_score:
                    max_score = score
                    max_score_data = (j, centro_x, centro_y, x1, y1, x2, y2)
                    max_centro=[centro_x,centro_y]
                 # Agregar los datos a la nueva lista con el puntaje
                data_list_score.append((j, centro_x, centro_y, x1, y1, x2, y2, area, score))
    
            # Ordenar data_list_score por puntaje
            data_list_score.sort(key=lambda x: x[8], reverse=True)
        
    
        
        # Salir del bucle al presionar la tecla 'Esc'
        # k = cv2.waitKey(20) & 0xFF
        # if k == 27:
        #     break
        
        
        # if mover==True:
        #     # Dibujar los puntos guardados en la lista
        for elemento in data_list_score:
            # Extraer los elementos [1] y [2]
            centro_x = elemento[1]
            centro_y = elemento[2]
    
            # Agregar una dupla a la lista puntos
            puntos.append((centro_x, centro_y))
            print(puntos)
            
        for punto in puntos:
            cv2.circle(frame, punto, 5, (255, 0, 0), -1)  
        puntos=[]
        # Imprimir el índice y el centro del objeto con el puntaje más alto
        if max_score_data is not None:
            max_index, max_centro_x, max_centro_y, max_x1, max_y1, max_x2, max_y2 = max_score_data
            print("Objeto con puntaje más alto:")
            print(f"Índice: {max_index}, Centro: ({max_centro_x}, {max_centro_y}), Puntaje: {max_score}")
        
        # Dibujar los bounding boxes en el frame
        contador = 1
        for j, centro_x, centro_y, x1, y1, x2, y2, area, score in data_list_score:
            # Calcular el tono del color azul
            tono_azul = int(255 - (contador * 255 / len(data_list_score)))
            tono_rojo=contador*255/len(data_list_score)
            color = (tono_azul, 0, tono_rojo)  # Color azul que se torna rojo
        
            # Dibujar el bounding box en el frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
            # Dibujar el centro del bounding box como un círculo
            cv2.circle(frame, (centro_x, centro_y), 5, (0, 255, 0), -1)
        
            # Dibujar el número de identificación del bounding box
            cv2.putText(frame, str(contador), (centro_x, centro_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        
            # Incrementar el contador
            contador += 1
        
       
                    
    
    #-----------------------------------------------------------------------------------------------        
        cv2.circle(frame, (320,240), 5, (0, 0, 255), -1)    
        #cv2.imshow('Webcam', frame)
        
        
        #ruta_archivo = guardar_frame_con_fecha(frame,1)
        
        frame_queue.put(frame)
            
        for elemento in data_list_score:
            # Extraer los elementos [1] y [2]
            centro_x = elemento[1]
            centro_y = elemento[2]
    
            # Agregar una dupla a la lista puntos
            puntos.append((centro_x, centro_y))
            
        if not puntos:
            print("Sin objetos")
        else:
            
            puntos_pwm=to_pwm(puntos)
            print("pwm_puntos: ", puntos_pwm)
            ultimate_puntos=pwms_C(puntos_pwm)
            print("pwm_puntos_relativos: ", ultimate_puntos)
        
            angles=to_angulos(puntos_pwm)
            d3=enviar_datos_por_serial(ultimate_puntos)
        
            radar = dibujar_radar(frame, angles,d3)
            #cv2.imshow('Radar', radar)
            ruta_archivo = guardar_frame_con_fecha(radar,0)
            puntos=[]
            ser.write(b'r')
        
        
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")      
finally:
#time.sleep(10)
# print(puntos_pwm)
# print(angles)
# print(data_list_score)
# print(puntos)


    cap.release()
        # Detener el hilo de grabación
    stop_event.set()
    grabar_thread.join()
    ser.write(b'r')
    ser.close()
    cv2.destroyAllWindows()

cap.release()
        # Detener el hilo de grabación
stop_event.set()
grabar_thread.join()
ser.write(b'r')
ser.close()
cv2.destroyAllWindows()
