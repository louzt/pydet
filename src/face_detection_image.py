"""
Detección de rostros en imágenes estáticas
"""
import cv2
import argparse
import sys
import os
from utils import load_face_cascade, detect_faces, draw_faces, save_faces, resize_image

def detect_faces_in_image(image_path, output_path=None, save_detected_faces=False):
    """
    Detecta rostros en una imagen y opcionalmente guarda el resultado
    
    Args:
        image_path (str): Ruta de la imagen de entrada
        output_path (str): Ruta donde guardar la imagen con rostros marcados
        save_detected_faces (bool): Si guardar los rostros detectados por separado
    """
    # Verificar que el archivo existe
    if not os.path.exists(image_path):
        print(f"Error: No se encontró la imagen en {image_path}")
        return
    
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde {image_path}")
        return
    
    # Redimensionar si es muy grande
    image = resize_image(image)
    
    # Cargar el clasificador
    try:
        face_cascade = load_face_cascade()
    except ValueError as e:
        print(f"Error al cargar el clasificador: {e}")
        return
    
    # Detectar rostros
    faces = detect_faces(image, face_cascade)
    
    print(f"Se detectaron {len(faces)} rostro(s) en la imagen")
    
    if len(faces) > 0:
        # Dibujar rectángulos alrededor de los rostros
        result_image = draw_faces(image.copy(), faces)
        
        # Mostrar la imagen con rostros detectados
        cv2.imshow('Rostros Detectados', result_image)
        
        # Guardar imagen si se especifica
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Imagen guardada en: {output_path}")
        
        # Guardar rostros individuales si se especifica
        if save_detected_faces:
            save_faces(image, faces)
        
        print("Presiona cualquier tecla para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No se detectaron rostros en la imagen")

def main():
    parser = argparse.ArgumentParser(description='Detección de rostros en imágenes')
    parser.add_argument('--image', '-i', required=True, 
                       help='Ruta de la imagen de entrada')
    parser.add_argument('--output', '-o', 
                       help='Ruta donde guardar la imagen con rostros detectados')
    parser.add_argument('--save-faces', action='store_true',
                       help='Guardar rostros detectados como imágenes separadas')
    
    args = parser.parse_args()
    
    detect_faces_in_image(args.image, args.output, args.save_faces)

if __name__ == "__main__":
    main()
