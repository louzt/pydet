"""
Utilidades comunes para detección de rostros
"""
import cv2
import os

def load_face_cascade(cascade_path=None):
    """
    Carga el clasificador Haar Cascade para detección de rostros
    
    Args:
        cascade_path (str): Ruta al archivo XML del clasificador
    
    Returns:
        cv2.CascadeClassifier: Clasificador cargado
    """
    if cascade_path is None:
        # Usar el clasificador incluido con OpenCV
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise ValueError(f"No se pudo cargar el clasificador desde: {cascade_path}")
    
    return face_cascade

def detect_faces(image, face_cascade, scale_factor=1.1, min_neighbors=5):
    """
    Detecta rostros en una imagen
    
    Args:
        image: Imagen en formato OpenCV
        face_cascade: Clasificador Haar Cascade
        scale_factor (float): Factor de escala para la detección
        min_neighbors (int): Número mínimo de vecinos requeridos
    
    Returns:
        list: Lista de coordenadas de los rostros detectados (x, y, w, h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=scale_factor, 
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    return faces

def draw_faces(image, faces, color=(255, 0, 0), thickness=2):
    """
    Dibuja rectángulos alrededor de los rostros detectados
    
    Args:
        image: Imagen en formato OpenCV
        faces: Lista de coordenadas de rostros (x, y, w, h)
        color (tuple): Color del rectángulo en formato BGR
        thickness (int): Grosor del rectángulo
    
    Returns:
        image: Imagen con los rectángulos dibujados
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image

def save_faces(image, faces, output_dir="detected_faces"):
    """
    Guarda los rostros detectados como imágenes separadas
    
    Args:
        image: Imagen original
        faces: Lista de coordenadas de rostros
        output_dir (str): Directorio donde guardar las imágenes
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, (x, y, w, h) in enumerate(faces):
        face_img = image[y:y+h, x:x+w]
        filename = os.path.join(output_dir, f"face_{i+1}.jpg")
        cv2.imwrite(filename, face_img)
        print(f"Rostro guardado: {filename}")

def resize_image(image, max_width=800):
    """
    Redimensiona la imagen manteniendo la proporción
    
    Args:
        image: Imagen en formato OpenCV
        max_width (int): Ancho máximo deseado
    
    Returns:
        image: Imagen redimensionada
    """
    height, width = image.shape[:2]
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        image = cv2.resize(image, (max_width, new_height))
    return image
