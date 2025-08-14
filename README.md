# Python Face Detection Project

Este proyecto implementa detección de rostros utilizando OpenCV y Python.

## Características

- Detección de rostros en imágenes
- Detección de rostros en video en tiempo real
- Detección de rostros desde la cámara web
- Guardado de rostros detectados

## Requisitos

- Python 3.7+
- OpenCV
- NumPy

## Instalación

1. Clona este repositorio
2. Crea un entorno virtual:
   ```bash
   python -m venv venv
   ```
3. Activa el entorno virtual:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Detección en imagen
```bash
python src/face_detection_image.py --image path/to/image.jpg
```

### Detección en video
```bash
python src/face_detection_video.py --video path/to/video.mp4
```

### Detección desde cámara
```bash
python src/face_detection_camera.py
```

## Estructura del Proyecto

```
pydet/
├── src/
│   ├── __init__.py
│   ├── face_detection_image.py
│   ├── face_detection_video.py
│   ├── face_detection_camera.py
│   └── utils.py
├── models/
│   └── haarcascade_frontalface_default.xml
├── examples/
│   └── sample_images/
├── requirements.txt
├── README.md
└── .gitignore
```

## Contribuir

1. Haz un fork del proyecto
2. Crea una rama para tu función (`git checkout -b feature/AmazingFeature`)
3. Haz commit de tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Haz push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT - mira el archivo [LICENSE](LICENSE) para más detalles.
