"""
Detección de rostros desde la cámara web en tiempo real
"""
import cv2
import argparse
from utils import load_face_cascade, detect_faces, draw_faces

def detect_faces_from_camera(camera_index=0, save_snapshots=False):
    """
    Detecta rostros desde la cámara web en tiempo real
    
    Args:
        camera_index (int): Índice de la cámara (0 para la cámara principal)
        save_snapshots (bool): Si guardar instantáneas cuando se detecten rostros
    """
    # Abrir la cámara
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara {camera_index}")
        return
    
    # Cargar el clasificador
    try:
        face_cascade = load_face_cascade()
    except ValueError as e:
        print(f"Error al cargar el clasificador: {e}")
        return
    
    # Configurar la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Detección de rostros en tiempo real iniciada...")
    print("Controles:")
    print("- Presiona 'q' para salir")
    print("- Presiona 's' para tomar una instantánea")
    print("- Presiona 'r' para alternar grabación")
    
    snapshot_count = 0
    recording = False
    out = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer frame de la cámara")
            break
        
        # Voltear horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Detectar rostros
        faces = detect_faces(frame, face_cascade)
        
        # Dibujar rectángulos alrededor de los rostros
        result_frame = draw_faces(frame, faces)
        
        # Agregar información en pantalla
        cv2.putText(result_frame, f'Rostros detectados: {len(faces)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Indicar si está grabando
        if recording:
            cv2.putText(result_frame, 'GRABANDO', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(result_frame, (550, 30), 10, (0, 0, 255), -1)
        
        # Mostrar controles
        cv2.putText(result_frame, "Presiona 'q' para salir, 's' para snapshot, 'r' para grabar", 
                   (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mostrar frame
        cv2.imshow('Detección de Rostros - Cámara Web', result_frame)
        
        # Escribir frame si está grabando
        if recording and out:
            out.write(result_frame)
        
        # Manejar teclas presionadas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s') and len(faces) > 0:
            # Tomar instantánea solo si hay rostros detectados
            snapshot_count += 1
            filename = f'snapshot_{snapshot_count}.jpg'
            cv2.imwrite(filename, result_frame)
            print(f"Instantánea guardada: {filename}")
        elif key == ord('r'):
            # Alternar grabación
            if not recording:
                # Iniciar grabación
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('recording.mp4', fourcc, 20.0, (640, 480))
                recording = True
                print("Grabación iniciada...")
            else:
                # Detener grabación
                if out:
                    out.release()
                    out = None
                recording = False
                print("Grabación detenida. Video guardado como: recording.mp4")
    
    # Limpiar recursos
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada")

def main():
    parser = argparse.ArgumentParser(description='Detección de rostros desde cámara web')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Índice de la cámara (default: 0)')
    parser.add_argument('--save-snapshots', action='store_true',
                       help='Habilitar guardado automático de instantáneas')
    
    args = parser.parse_args()
    
    detect_faces_from_camera(args.camera, args.save_snapshots)

if __name__ == "__main__":
    main()
