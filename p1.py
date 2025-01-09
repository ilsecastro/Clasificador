import os
import cv2
import shutil
from ultralytics import YOLO
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tkinter import Frame, Tk, filedialog, Label, Button, Toplevel
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
    imagenes_un_objeto = {}
    for archivo in os.listdir(dataset_path):
        if archivo.endswith((".jpg", ".png")):
            ruta_imagen = os.path.join(dataset_path, archivo)
            imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
            
            # Verificar si la imagen se cargó correctamente
            if imagen is None:
                print(f"No se pudo cargar la imagen: {archivo}")
                continue
            
            # Aplicar umbral para binarizar la imagen
            _, imagen_binaria = cv2.threshold(imagen, 128, 255, cv2.THRESH_BINARY)
            
            # Detectar contornos
            contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos pequeños por área
            contornos = [c for c in contornos if cv2.contourArea(c) > 100]
            print(f"Imagen: {archivo}, Número de contornos detectados: {len(contornos)}")
            
            # Si hay exactamente un contorno significativo, consideramos que hay un objeto único
            if len(contornos) == 1:
                imagenes_un_objeto[archivo] = ruta_imagen
    return imagenes_un_objeto


def guardar_imagenes_un_objeto(imagenes_un_objeto, output_folder="imagenes_un_objeto"):
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for imagen_nombre, ruta in imagenes_un_objeto.items():
        # Mover cada imagen a la carpeta de salida
        destino = os.path.join(output_folder, imagen_nombre)
        shutil.copy(ruta, destino)  # Copiar la imagen en lugar de moverla
        print(f"Imagen {imagen_nombre} guardada en {output_folder}")

# Ejecutar la función con las imágenes identificadas
imagenes_un_objeto_identificadas = {
    "1.png": r"C:\Users\Lenovo\OneDrive\Documentos\Visión artficial\proyecto\dataset\1.png",
    "6.png": r"C:\Users\Lenovo\OneDrive\Documentos\Visión artficial\proyecto\dataset\6.png",
    "11.png": r"C:\Users\Lenovo\OneDrive\Documentos\Visión artficial\proyecto\dataset\11.png",
    "16.png": r"C:\Users\Lenovo\OneDrive\Documentos\Visión artficial\proyecto\dataset\16.png",
    "21.png": r"C:\Users\Lenovo\OneDrive\Documentos\Visión artficial\proyecto\dataset\21.png",
}

guardar_imagenes_un_objeto(imagenes_un_objeto_identificadas)

# Mostrar imágenes de objetos únicos
def mostrar_objetos_unicos_filtrados(imagenes_un_objeto_identificadas):
    ventana_unicos = Toplevel()  # Crear una nueva ventana
    ventana_unicos.title("Objetos Únicos en el Dataset")
    
    for i, (nombre, ruta) in enumerate(imagenes_un_objeto_identificadas.items()):
        img = Image.open(ruta)
        img.thumbnail((150, 150))  # Redimensionar la imagen

        # Convertir la imagen para que sea compatible con Tkinter
        img_tk = ImageTk.PhotoImage(img)

        label = Label(ventana_unicos, text=f"{nombre}", compound="top", image=img_tk)
        label.image = img_tk  # Mantener referencia
        label.grid(row=0, column=i)
mostrar_objetos_unicos_filtrados(imagenes_un_objeto_identificadas)        



# Paso 2: Extracción de características
def extraer_caracteristicas(dataset_path):
    caracteristicas = []
    imagenes_nombres = []
    for imagen_nombre in os.listdir(dataset_path):
        imagen_path = os.path.join(dataset_path, imagen_nombre)
        if os.path.isfile(imagen_path):
            imagen = cv2.imread(imagen_path)
            if imagen is None:
                print(f"No se pudo cargar la imagen: {imagen_nombre}")
                continue
            imagen_redimensionada = cv2.resize(imagen, (224, 224))  # Ajusta el tamaño
            imagen_gris = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2GRAY)
            vector = imagen_gris.flatten()  # Convertir a vector
            caracteristicas.append(vector)
            imagenes_nombres.append(imagen_nombre)

    # Convertir los nombres de las imágenes a nombres relativos
    nombres = [os.path.basename(imagen_nombre) for imagen_nombre in imagenes_nombres]

     # Validar datos antes de PCA
    if not caracteristicas or len(caracteristicas) < 2:
        raise ValueError("No hay suficientes características para realizar PCA.")  
    n_components = min(len(caracteristicas), len(caracteristicas[0]), 100)
    print(f"Reduciendo a {n_components} componentes.")
    pca = PCA(n_components=n_components)
    caracteristicas_reducidas = pca.fit_transform(caracteristicas)

    return caracteristicas_reducidas, nombres



# Paso 3: Clasificación
def entrenar_clasificador(caracteristicas, etiquetas):
    clasificador = SVC(kernel='linear', probability=True)
    clasificador.fit(caracteristicas, etiquetas)
    return clasificador


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
 
    


# Paso 4: Clasificar una imagen y mostrar las más similares
def clasificar_y_mostrar_similares(clasificador, caracteristicas, nombres_imagenes, imagen_seleccionada, dataset_path):
    # Obtener el índice de la imagen seleccionada
    nombres_imagenes = list(nombres_imagenes)  # Convertir a lista
    idx_seleccionada = nombres_imagenes.index(imagen_seleccionada)
    vector_imagen = caracteristicas[idx_seleccionada]
    assert vector_imagen.shape[0] == caracteristicas.shape[1], "Las dimensiones de los datos no coinciden."

    # Calcular similitudes con todas las imágenes
    similitudes = cosine_similarity([vector_imagen], caracteristicas)[0]
    # Calcular similitudes con todas las imágenes
    probabilidades = clasificador.decision_function(caracteristicas)
    print(f"Dimensiones de probabilidades: {probabilidades.shape}")
    print(f"Dimensiones de vector_imagen: {vector_imagen.shape}")

    # Obtener los índices de las 5 imágenes más similares
    indices_similares = np.argsort(similitudes)[-5:][::-1]
    # Devolver las 5 imágenes más similares
    imagenes_similares = [nombres_imagenes[i] for i in indices_similares]
    # Mostrar las imágenes similares en la interfaz gráfica
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
    dataset_path = cargar_dataset()
    if not dataset_path:
        return  
    
    # Verificar el dataset seleccionado
    print(f"Dataset seleccionado: {dataset_path}")
    
    # Filtrar imágenes con un único objeto
    print("Buscando imágenes con un único objeto...")
    imagenes_un_objeto = filtrar_imagenes_con_un_objeto(dataset_path)
    
    if not imagenes_un_objeto:
        print("No se encontraron imágenes con un único objeto.")
        return

    # Mostrar las imágenes de objetos únicos
    print("Mostrando imágenes con un único objeto...")
    mostrar_objetos_unicos_filtrados(imagenes_un_objeto)



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
