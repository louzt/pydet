"""
Script de prueba para verificar la instalación y funcionamiento básico
"""
import cv2
import numpy as np
import sys
import os

def test_opencv_installation():
    """Prueba que OpenCV esté correctamente instalado"""
    print("=== Prueba de Instalación de OpenCV ===")
    print(f"Versión de OpenCV: {cv2.__version__}")
    print(f"Versión de NumPy: {np.__version__}")
    
    # Verificar que se puede cargar el clasificador Haar
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print("❌ Error: No se pudo cargar el clasificador Haar")
            return False
        else:
            print("✅ Clasificador Haar cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar clasificador: {e}")
        return False
    
    # Crear una imagen de prueba
    try:
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        test_image[:] = (100, 150, 200)  # Color de fondo
        
        # Intentar detectar rostros (no debería encontrar ninguno)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(f"✅ Detección de rostros funcional (rostros encontrados: {len(faces)})")
        
    except Exception as e:
        print(f"❌ Error en detección de rostros: {e}")
        return False
    
    print("\n=== Prueba de Cámara ===")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Cámara detectada y accesible")
            cap.release()
        else:
            print("⚠️  Cámara no disponible (esto es normal si no hay cámara conectada)")
    except Exception as e:
        print(f"⚠️  Error al acceder a la cámara: {e}")
    
    print("\n=== Resumen ===")
    print("✅ OpenCV instalado correctamente")
    print("✅ Sistema listo para detección de rostros")
    print("\nPara probar con imágenes:")
    print("1. Coloca una imagen en examples/sample_images/")
    print("2. Ejecuta: python src/face_detection_image.py --image examples/sample_images/tu_imagen.jpg")
    print("\nPara probar con cámara:")
    print("1. Ejecuta: python src/face_detection_camera.py")
    
    return True

if __name__ == "__main__":
    success = test_opencv_installation()
    if success:
        print("\n🎉 ¡Instalación exitosa!")
        sys.exit(0)
    else:
        print("\n❌ Problemas en la instalación")
        sys.exit(1)
