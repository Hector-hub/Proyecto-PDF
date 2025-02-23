# Procesador de PDFs

Este proyecto procesa archivos PDF (como publicaciones de información aeronáutica) para extraer texto, imágenes y generar embeddings usando modelos de aprendizaje profundo y procesamiento de lenguaje natural.

## Descripción

El propósito principal es analizar PDFs, extraer contenido y clasificarlo utilizando técnicas híbridas que combinan modelos como CLIP y clasificadores especializados.

## Características Principales

- 📄 **Procesamiento de PDFs**  
  Extracción estructurada de texto e imágenes de documentos técnicos
- 🖼️ **Clasificación Híbrida de Imágenes**  
  Combina modelos CLIP y especializados para identificación precisa de diagramas técnicos
- 🔍 **Detección de Objetos**  
  Identificación de elementos aeronáuticos usando YOLOv8
- 📊 **Generación de Embeddings**  
  Creación de representaciones vectoriales usando modelos de OpenAI
- ✨ **OCR Optimizado**  
  Reconocimiento de texto en imágenes con ajustes para documentos técnicos
- 🧠 **Modelos Especializados**  
  Soporte para clasificación de cartas de navegación, displays radar y diagramas de aeropuertos

## Requisitos

- **Python**: 3.8 o superior
- **Dependencias**:
  - `fitz` (PyMuPDF)
  - `pymupdf`
  - `pytest`
  - `pytesseract`
  - `torch`
  - `clip`
  - `PIL` (Pillow)
  - `transformers`
  - `sentence-transformers`
  - `ultralytics`
  - `torchvision`
  - `openai`
  - `tiktoken`
- Instala las dependencias con:
  ```bash
  pip install -r requirements.txt
  ```

## Configuración del Entorno

1. **Clona el Repositorio**:

   ```bash
   git clone https://github.com/tu_usuario/tu_repositorio.git
   cd tu_repositorio
   ```

2. **Configura las Rutas de Directorio**:

   - Las variables `BASE_DIR` y `INPUT_PDFS` en `process_pdf.py` son específicas del sistema del usuario. Debes modificarlas para que apunten a tus propios directorios.
   - Ejemplo original (no funcionará en tu máquina):
     ```python
     BASE_DIR = "/Users/hecrey/Desktop/PDF_NLP_Project/process_pdf"
     INPUT_PDFS = "/Users/hecrey/Desktop/PDF_NLP_Project/input_pdfs"
     ```
   - **Cómo ajustarlo**:
     1. Crea una carpeta para el proyecto, por ejemplo, `/home/tu_usuario/PDF_NLP_Project`.
     2. Dentro, crea un subdirectorio `input_pdfs` para los PDFs de entrada.
     3. Actualiza las variables en `process_pdf.py`:
        ```python
        BASE_DIR = "/home/tu_usuario/PDF_NLP_Project/process_pdf"
        INPUT_PDFS = "/home/tu_usuario/PDF_NLP_Project/input_pdfs"
        ```

3. **Configura la API de OpenAI**:
   - Obtén una clave API de OpenAI y configúrala en el cliente en la raiz del proyecto.
   ```
   export OPENAI_API_KEY="Your apikey"
   ```

## Ejecución del Código

1. Coloca los PDFs a procesar en el directorio configurado en `INPUT_PDFS`.
2. Ejecuta el script principal:
   ```bash
   python process_pdf.py
   ```
   or
   ```
   python3 process_pdf.py
   ```
3. Los resultados (texto extraído, imágenes procesadas y embeddings) se guardarán en un subdirectorio como `pdf_processed`.

## Ejecución de Pruebas Unitarias

1. Asegúrate de que las pruebas estén en el directorio `tests/`.
2. Ejecuta todas las pruebas:
   ```bash
   python -m unittest discover -s tests
   ```
   or
   ```
   python3 -m unittest discover -s tests
   ```
3. Las pruebas verifican funciones clave como la generación de embeddings, clasificación y procesamiento de PDFs.

## Colaboración

- **Reportar Problemas**: Usa [GitHub Issues](https://github.com/tu_usuario/tu_repositorio/issues).
- **Contribuir**:
  1. Haz un fork del repositorio.
  2. Crea una rama para tu cambio: `git checkout -b mi-cambio`.
  3. Realiza tus modificaciones y haz commit.
  4. Envía un pull request al repositorio original.
- **Convenciones**: Usa PEP 8 para el estilo de código.

## Licencia

Este proyecto está bajo la [Licencia MIT](LICENSE).

## Notas Adicionales

- Si tienes problemas con la libreria `clip` ejecuta el siguiente comando:

```
pip uninstall clip-by-openai -y
pip install git+https://github.com/openai/CLIP.git
```

- Si encuentras errores relacionados con rutas, revisa que `BASE_DIR` y `INPUT_PDFS` estén correctamente configurados.
- Asegúrate de tener suficiente espacio en disco, ya que el procesamiento de PDFs puede generar archivos grandes.
