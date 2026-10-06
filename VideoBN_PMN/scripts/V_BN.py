import cv2                      # 1- Uso de OpenCV para lectura y escritura de videos
import numpy as np             # Librería para manejo de arreglos y matrices
from PIL import Image          # 1- Uso de Pillow para manipulación de imágenes
import time                    # 3- Medición del tiempo total de ejecución
import os                      # Permite obtener los hilos disponibles del sistema

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
def procesar_frame(frame):
    # 1- Uso de Pillow
    # Conversión de formato OpenCV (BGR) a Pillow (RGB)
    pil_img = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    # 2- Conversión de cada fotograma a blanco y negro
    pil_bn = pil_img.convert("L")

    # Conversión nuevamente a formato OpenCV
    frame_bn = cv2.cvtColor(
        np.array(pil_bn),
        cv2.COLOR_GRAY2BGR
    )
    return frame_bn
def main():
    # Ruta del video original
    ruta_entrada = "../Video/video_color.mp4"
    # Video final generado
    ruta_salida = "../resultados/video_bnpmn.mp4"
    # 1- Uso de OpenCV para abrir el video
    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return
    # 4- Obtención de propiedades del video
    fps_real = cap.get(cv2.CAP_PROP_FPS)
    # Redondea valores como 59.94 a 60 fps
    fps = round(fps_real)
    # 4- Obtención de resolución
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 4- Cálculo de duración del video
    duracion = total_frames / fps
    print(f"Duración del video: {duracion/60:.2f} minutos")
    print(f"Resolución: {ancho}x{alto} a {fps} fps")
    # Uso de OpenCV para generar el video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        ruta_salida,
        fourcc,
        fps,
        (ancho, alto)
    )
    # 3- Inicio de medición del tiempo total de ejecución
    inicio = time.time()
    # Cantidad de fotogramas enviados por lote
    tam_lote = 50
    # Obtiene los hilos lógicos disponibles del sistema
    nucleos_disponibles = os.cpu_count()
    # 16 hilos a 32 procesos
    procesos_utilizados = nucleos_disponibles * 2

    print(f"Hilos del sistema: {nucleos_disponibles}")
    print(f"Procesos configurados: {procesos_utilizados}")
    with ProcessPoolExecutor(
            max_workers=procesos_utilizados
    ) as executor:
        while True:
            lote = []
            # 2- Lectura de fotogramas del video
            # Se leen uno por uno desde el archivo original
            for _ in range(tam_lote):
                ret, frame = cap.read()
                if not ret:
                    break
                lote.append(frame)
            if len(lote) == 0:
                break
            resultados = executor.map(
                procesar_frame,
                lote
            )
            # Guardar cada fotograma convertido
            for frame_bn in resultados:
                out.write(frame_bn)
    # Liberación de recursos de OpenCV
    cap.release()
    out.release()
    # 3- Fin de la medición del tiempo total
    fin = time.time()
    tiempo_total = fin - inicio
    print(
        f"Tiempo total: {tiempo_total:.2f} segundos "
        f"({tiempo_total/60:.2f} minutos)"
    )
    print(f"Video generado en: {ruta_salida}")
if __name__ == "__main__":
    # Necesario para multiprocessing en Windows
    freeze_support()

    main()