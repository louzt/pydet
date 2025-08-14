"""
Detección de rostros en video
"""
import cv2
import argparse
import sys
import os
from utils import load_face_cascade, detect_faces, draw_faces

def detect_faces_in_video(video_path, output_path=None):
    """
    Detecta rostros en un video y opcionalmente guarda el resultado
    
    Args:
        video_path (str): Ruta del video de entrada
        output_path (str): Ruta donde guardar el video procesado
    """
    # Verificar que el archivo existe
    if not os.path.exists(video_path):
        print(f"Error: No se encontró el video en {video_path}")
        return
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return
    
    # Cargar el clasificador
    try:
        face_cascade = load_face_cascade()
    except ValueError as e:
        print(f"Error al cargar el clasificador: {e}")
        return
    
    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Configurar escritor de video si se especifica salida
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Procesando video... Presiona 'q' para salir")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detectar rostros
        faces = detect_faces(frame, face_cascade)
        
        # Dibujar rectángulos alrededor de los rostros
        result_frame = draw_faces(frame, faces)
        
        # Mostrar información
        cv2.putText(result_frame, f'Rostros: {len(faces)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_frame, f'Frame: {frame_count}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow('Detección de Rostros en Video', result_frame)
        
        # Escribir frame al video de salida
        if out:
            out.write(result_frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Limpiar
    cap.release()
    if out:
        out.release()
        print(f"Video procesado guardado en: {output_path}")
    cv2.destroyAllWindows()
    
    print(f"Procesamiento completado. Total de frames: {frame_count}")

def main():
    parser = argparse.ArgumentParser(description='Detección de rostros en video')
    parser.add_argument('--video', '-v', required=True, 
                       help='Ruta del video de entrada')
    parser.add_argument('--output', '-o', 
                       help='Ruta donde guardar el video procesado')
    
    args = parser.parse_args()
    
    detect_faces_in_video(args.video, args.output)

if __name__ == "__main__":
    main()
