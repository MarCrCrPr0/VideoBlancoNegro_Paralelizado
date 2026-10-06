
import cv2                      # 1- Uso de OpenCV para leer y escribir videos
import numpy as np             # Librería para manejar matrices de píxeles
from PIL import Image          # 1- Uso de Pillow para convertir imágenes a blanco y negro
import time                    # 3- Uso para medir el tiempo total de procesamiento

# Librerías para procesamiento paralelo
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

def procesar_frame(frame):
    # 1- Uso de Pillow
    # Convierte el fotograma de OpenCV (BGR)
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
    # Ruta del video generado
    ruta_salida = "../resultados/video_bn2n.mp4"
    # Abrir video con OpenCV
    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return
    # 4- Obtención de duración, resolución y FPS
    fps_real = cap.get(cv2.CAP_PROP_FPS)
    fps = round(fps_real)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 4- Cálculo de duración del video
    duracion = total_frames / fps

    print(f"Duración del video: {duracion/60:.2f} minutos")
    print(f"Resolución: {ancho}x{alto} a {fps} fps")

    # Crear archivo de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        ruta_salida,
        fourcc,
        fps,
        (ancho, alto)
    )

    # 3- Inicio de medición del tiempo total
    inicio = time.time()

    # Cantidad de fotogramas que se enviarán por lote
    tam_lote = 5
    with ProcessPoolExecutor(max_workers=2) as executor:
        while True:
            lote = []
            # 2- Lectura de fotogramas del video
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
    # Liberar recursos
    cap.release()
    out.release()
    # 3- Fin de medición del tiempo de ejecución
    fin = time.time()
    print(
        f"Tiempo total: {(fin-inicio):.2f} segundos "
        f"({(fin-inicio)/60:.2f} minutos)"
    )
if __name__ == "__main__":
    # Necesario para multiprocessing en Windows
    freeze_support()
    main()