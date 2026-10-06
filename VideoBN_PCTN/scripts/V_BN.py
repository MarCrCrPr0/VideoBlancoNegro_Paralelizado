import cv2                      # 1- Uso de OpenCV
import numpy as np             # Manejo de matrices
from PIL import Image          # 1- Uso de Pillow
import time                    # 3- Medición de tiempo
import os                      # Obtiene los núcleos disponibles

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

def procesar_frame(frame):
    # 1- Uso de Pillow
    # Conversión de OpenCV (BGR) a Pillow (RGB)
    pil_img = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    # 2- Conversión del fotograma a blanco y negro
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
    # Nuevo nombre
    ruta_salida = "../resultados/video_bnpctn.mp4"
    # Abrir video
    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return
    # 4- Obtener FPS, resolución y duración
    fps_real = cap.get(cv2.CAP_PROP_FPS)
    # Redondea valores como 59.94 a 60
    fps = round(fps_real)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion = total_frames / fps
    # 4- Mostrar duración y resolución
    print(f"Duración del video: {duracion/60:.2f} minutos")
    print(f"Resolución: {ancho}x{alto} a {fps} fps")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        ruta_salida,
        fourcc,
        fps,
        (ancho, alto)
    )
    # 3- Inicio de medición del tiempo
    inicio = time.time()
    # Cantidad de fotogramas por lote
    tam_lote = 50
    # Obtiene automáticamente todos los núcleos/hilos disponibles
    nucleos_disponibles = os.cpu_count()
    print(f"Núcleos disponibles: {nucleos_disponibles}")
    with ProcessPoolExecutor(
            max_workers=nucleos_disponibles
    ) as executor:
        while True:
            lote = []
            # 2- Lectura de fotogramas
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
            # Guardar fotogramas procesados
            for frame_bn in resultados:
                out.write(frame_bn)
    cap.release()
    out.release()
    # 3- Fin de medición del tiempo
    fin = time.time()
    tiempo_total = fin - inicio
    print(
        f"Tiempo total: {tiempo_total:.2f} segundos "
        f"({tiempo_total/60:.2f} minutos)"
    )
    print(f"Video generado en: {ruta_salida}")
if __name__ == "__main__":
    freeze_support()
    main()
    