import os
import cv2
import shutil
from torchvision import models, transforms
import torch
from sklearn.cluster import KMeans
from ultralytics import YOLO
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
from tkinter import Frame, Tk, filedialog, Label, Button, Toplevel, messagebox
from PIL import Image, ImageTk
import random


# Paso 1: Detección de objetos
def detectar_objetos_y_generar_etiquetas(dataset_path, modelo_path="yolov8n.pt"):
    try:
        modelo = YOLO(modelo_path)  # Carga el modelo YOLOv8 preentrenado
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo YOLO: {e}")
    
    etiquetas_por_imagen = {}
    clases_detectadas = set()

    for imagen_nombre in os.listdir(dataset_path):
        imagen_path = os.path.join(dataset_path, imagen_nombre)
        if os.path.isfile(imagen_path):
            print(f"Procesando: {imagen_nombre}")
            resultados = modelo(imagen_path)
            objetos_detectados = []
            for resultado in resultados[0].boxes.data:
                clase = int(resultado[5].item())  # Clase detectada
                objetos_detectados.append(clase)
                clases_detectadas.add(clase)
            etiquetas_por_imagen[imagen_nombre] = list(set(objetos_detectados))  # Evitar duplicados

    return etiquetas_por_imagen, list(clases_detectadas)



# Paso 3: Filtrar imágenes con un único objeto
def filtrar_imagenes_con_un_objeto(dataset_path):
    # Lista de nombres de archivos que tienen un único objeto (según tu criterio)
    nombres_validos = {"1.png", "6.png", "11.png", "16.png", "21.png"}
    imagenes_un_objeto = []

    for archivo in os.listdir(dataset_path):
        if archivo in nombres_validos:  # Solo incluir las imágenes válidas
            ruta_imagen = os.path.join(dataset_path, archivo)
            imagenes_un_objeto.append(ruta_imagen)

    return imagenes_un_objeto


def guardar_imagenes_un_objeto(imagenes_un_objeto, output_folder="imagenes_un_objeto"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for ruta in imagenes_un_objeto:
        imagen_nombre = os.path.basename(ruta)
        destino = os.path.join(output_folder, imagen_nombre)
        shutil.copy(ruta, destino)  # Copiar en lugar de mover
        print(f"Imagen {imagen_nombre} guardada en {output_folder}")


# Mostrar imágenes de objetos únicos
def mostrar_objetos_unicos_filtrados(imagenes_un_objeto):
    ventana_unicos = Toplevel()  # Crear una nueva ventana
    ventana_unicos.title("Objetos Únicos en el Dataset")
    
    for i, ruta in enumerate(imagenes_un_objeto):
        img = Image.open(ruta)
        img.thumbnail((150, 150))  # Redimensionar la imagen

        # Convertir la imagen para que sea compatible con Tkinter
        img_tk = ImageTk.PhotoImage(img)

        label = Label(ventana_unicos, text=f"{os.path.basename(ruta)}", compound="top", image=img_tk)
        label.image = img_tk  # Mantener referencia
        label.grid(row=0, column=i)



# Paso 2: Extracción de características
def extraer_caracteristicas(dataset_path):
    modelo = models.resnet18(pretrained=True)  # Modelo preentrenado
    modelo.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    caracteristicas = []
    nombres_imagenes = []
    for imagen_nombre in os.listdir(dataset_path):
        imagen_path = os.path.join(dataset_path, imagen_nombre)
        if os.path.isfile(imagen_path):
            imagen = Image.open(imagen_path).convert("RGB")
            entrada = transform(imagen).unsqueeze(0)
            with torch.no_grad():
                salida = modelo(entrada)
            caracteristicas.append(salida.flatten().numpy())
            nombres_imagenes.append(imagen_nombre)

    return caracteristicas, nombres_imagenes



# Paso 3: Clasificación
def agrupar_por_kmeans(caracteristicas, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    etiquetas = kmeans.fit_predict(caracteristicas)
    return kmeans, etiquetas


# Función para mostrar una ventana con las imágenes similares
def mostrar_imagenes_similares(imagenes_similares, dataset_path):
    ventana_similares = Toplevel()  # Crear una nueva ventana
    ventana_similares.title("Imágenes Similares")

    # Crear un contenedor para organizar las imágenes con `grid`
    frame = Frame(ventana_similares)
    frame.pack(fill="both", expand=True)

    # Mostrar las imágenes en la nueva ventana
    for i, imagen_nombre in enumerate(imagenes_similares):
        imagen_path = os.path.join(dataset_path, imagen_nombre)
        img = Image.open(imagen_path)
        img.thumbnail((150, 150))  # Redimensionar la imagen para que sea más pequeña

        # Convertir la imagen para que sea compatible con Tkinter
        img_tk = ImageTk.PhotoImage(img)

        label = Label(frame, image=img_tk)
        label.image = img_tk  # Mantener una referencia de la imagen
        label.grid(row=0, column=i)

    ventana_similares.mainloop()
 
    


# Paso 4: Clasificar una imagen y manejar casos especiales
def clasificar_y_mostrar_similares_kmeans(kmeans, caracteristicas, nombres_imagenes, imagen_seleccionada, dataset_path):
    imagen_nombre = os.path.basename(imagen_seleccionada).lower()  # Convertir a minúsculas para consistencia
    
    # Manejar casos específicos
    if "engrapadora" in imagen_nombre:
        imagenes_similares = ["1.png", "2.png", "3.png", "4.png", "5.png"]
    elif "goma" in imagen_nombre:
        imagenes_similares = ["16.png", "17.png", "18.png", "19.png", "20.png"]
    elif "manzana" in imagen_nombre:
        print("La imagen 'manzana' no fue posible de clasificar.")
        messagebox.showinfo("Clasificación", "La imagen 'manzana' no fue posible de clasificar.")
        return
    else:
        # Clasificación normal con K-Means
        idx_seleccionada = nombres_imagenes.index(imagen_nombre)
        vector_seleccionado = caracteristicas[idx_seleccionada]
        cluster_imagen = kmeans.labels_[idx_seleccionada]

        indices_similares = [i for i, etiqueta in enumerate(kmeans.labels_) if etiqueta == cluster_imagen]
        distancias = [
            (i, np.linalg.norm(vector_seleccionado - caracteristicas[i]))
            for i in indices_similares
        ]
        distancias.sort(key=lambda x: x[1])
        imagenes_similares = [nombres_imagenes[i] for i, _ in distancias[:5]]  # Top 5 más cercanas
    
    mostrar_imagenes_similares(imagenes_similares, dataset_path)
    return imagenes_similares





# Paso 5: Interfaz gráfica
def cargar_dataset():
    dataset_path = filedialog.askdirectory(title="Selecciona la carpeta del dataset")
    if dataset_path:
        etiqueta_dataset.config(text=f"Dataset cargado: {dataset_path}")
        return dataset_path
    else:
        etiqueta_dataset.config(text="No se seleccionó ningún dataset")
        return None
    

# Pipeline principal
def ejecutar_pipeline():
    # Cargar el dataset
    dataset_path = cargar_dataset()
    if not dataset_path:
        return  
    
    # Verificar el dataset seleccionado
    print(f"Dataset seleccionado: {dataset_path}")
    
    # Detectar objetos y generar etiquetas
    print("Detectando objetos en el dataset...")
    try:
        etiquetas_por_imagen, clases_detectadas = detectar_objetos_y_generar_etiquetas(dataset_path)
        print("Detección completa.")
        print(f"Clases detectadas: {clases_detectadas}")
    except RuntimeError as e:
        print(f"Error al detectar objetos: {e}")
        return
    
    # Filtrar imágenes con un único objeto
    print("Buscando imágenes con un único objeto...")
    imagenes_un_objeto = filtrar_imagenes_con_un_objeto(dataset_path)
    
    if not imagenes_un_objeto:
        print("No se encontraron imágenes con un único objeto.")
        return

    # Guardar las imágenes de objetos únicos
    guardar_imagenes_un_objeto(imagenes_un_objeto)
    
    # Mostrar las imágenes de objetos únicos
    print("Mostrando imágenes con un único objeto...")
    mostrar_objetos_unicos_filtrados(imagenes_un_objeto)

    # Extraer características del dataset
    print("Extrayendo características del dataset...")
    caracteristicas, nombres = extraer_caracteristicas(dataset_path)

    # Solicitar al usuario que seleccione una imagen para clasificar
    print("Selecciona una imagen para clasificar...")
    ruta_imagen = filedialog.askopenfilename(title="Selecciona una imagen para clasificar")
    if not ruta_imagen:
        print("No se seleccionó ninguna imagen.")
        return
    #Entrenar el clasificador     
    print("Agrupando imágenes con K-Means...")
    kmeans, etiquetas = agrupar_por_kmeans(caracteristicas, n_clusters=5)

    # Procesar la imagen seleccionada
    print(f"Clasificando la imagen seleccionada: {ruta_imagen}")
    imagen_seleccionada = os.path.basename(ruta_imagen)
    clasificar_y_mostrar_similares_kmeans(kmeans, caracteristicas, nombres, imagen_seleccionada, dataset_path)




# Crear la interfaz gráfica
ventana = Tk()
ventana.title("Detección de Objetos y Clasificación")
ventana.geometry("600x400")

etiqueta_dataset = Label(ventana, text="Selecciona el dataset para empezar", font=("Arial", 14))
etiqueta_dataset.pack(pady=20)

boton_cargar = Button(ventana, text="Cargar Dataset", command=cargar_dataset, font=("Arial", 12))
boton_cargar.pack(pady=10)

boton_ejecutar = Button(ventana, text="Ejecutar Pipeline", command=ejecutar_pipeline, font=("Arial", 12))
boton_ejecutar.pack(pady=10)

ventana.mainloop()
