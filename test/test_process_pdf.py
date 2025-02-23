import unittest
from unittest.mock import patch, Mock, mock_open, call
import os
import json
import torch
from process_pdf.process_pdf import (
    get_embedding, check_list_type, HybridClassifier, _format_detections,
    process_pdf, process_all_pdfs
)

class TestProcessPDF(unittest.TestCase):

    # Pruebas unitarias documentadas para casos de uso:
    
    #CASO DE USO #1 
    #Cuando los embeddings están dentro de una lista de listas. 
    def test_check_list_type_nested(self):
        """Prueba que check_list_type aplana una lista anidada."""
        embedding = [[0.1, 0.2, 0.3]]
        result = check_list_type(embedding)
        self.assertEqual(result, [0.1, 0.2, 0.3], "Debería aplanar la lista correctamente")

    #CASO DE USO #2 
    #Si no hay embeddings disponibles, la función debe devolver una lista vacía sin errores.
    def test_check_list_type_empty(self):
        """Prueba que check_list_type maneja una lista vacía."""
        embedding = []
        result = check_list_type(embedding)
        self.assertEqual(result, [], "Debería devolver una lista vacía sin cambios")
        
    #CASO DE USO #3
    #Si los embeddings ya están formateados, la función no debe alterarlos.
    def test_check_list_type_flat(self):
        """Prueba que check_list_type no modifica una lista plana."""
        embedding = [0.1, 0.2, 0.3]
        result = check_list_type(embedding)
        self.assertEqual(result, embedding, "Una lista plana debería devolverse sin cambios")

    #CASO DE USO #4 Procesamiento automático de todos los PDFs en una carpeta.
    def test_process_all_pdfs(self):
        """Prueba el procesamiento de múltiples PDFs."""
        with patch('os.listdir', return_value=["test1.pdf", "test2.pdf"]), \
             patch('process_pdf.process_pdf.process_pdf', return_value=("main.json", "embedding.json")) as mock_process:
            process_all_pdfs()
            self.assertEqual(mock_process.call_count, 2)
            mock_process.assert_has_calls([
                call(os.path.join("/Users/hecrey/Desktop/PDF_NLP_Project/input_pdfs", "test1.pdf")),
                call(os.path.join("/Users/hecrey/Desktop/PDF_NLP_Project/input_pdfs", "test2.pdf"))
            ])
            
    #CASO DE USO #5 
    #Si un usuario intenta procesar un archivo inexistente, el sistema debe continuar sin fallar.
    def test_process_all_pdfs_with_error(self):
        """Prueba el manejo de errores en process_all_pdfs."""
        with patch('os.listdir', return_value=["test.pdf"]), \
             patch('process_pdf.process_pdf.process_pdf', side_effect=Exception("no such file: '/Users/hecrey/Desktop/PDF_NLP_Project/input_pdfs/test.pdf'")) as mock_process, \
             patch('builtins.print') as mock_print:
            process_all_pdfs()
            mock_print.assert_called_with("❌ Error procesando test.pdf: no such file: '/Users/hecrey/Desktop/PDF_NLP_Project/input_pdfs/test.pdf'")


    #Otras pruebas unitarias:
    
    def test_process_pdf_error(self):
        """Prueba el manejo de errores al procesar un PDF corrupto."""
        with patch('fitz.open', side_effect=Exception("PDF corrupto")):
            with self.assertRaises(Exception, msg="Debería propagar errores al abrir el PDF"):
                process_pdf("corrupted.pdf")

    def setUp(self):
        """Configura un HybridClassifier antes de cada test."""
        self.classifier = HybridClassifier()

    def test_load_model_valid(self):
        """Prueba que load_model carga un modelo válido."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor, \
             patch('transformers.AutoModelForImageClassification.from_pretrained') as mock_model:
            mock_processor.return_value = "mock_processor"
            mock_model.return_value.to.return_value = "mock_model"
            result = self.classifier.load_model("chart")
            self.assertIn("chart", self.classifier.specialized_models)
            self.assertEqual(result["processor"], "mock_processor")
            self.assertEqual(result["model"], "mock_model")

    def test_load_model_invalid(self):
        """Prueba que load_model lanza error para modelo no soportado."""
        with self.assertRaises(ValueError, msg="Debería lanzar ValueError para tipo no soportado"):
            self.classifier.load_model("invalid_type")

    def test_needs_specialized(self):
        """Prueba la lógica para determinar si se necesita clasificación especializada."""
        self.assertEqual(self.classifier._needs_specialized("Aeronautical navigation chart"), "chart")
        self.assertEqual(self.classifier._needs_specialized("Air traffic control radar"), "radar")
        self.assertEqual(self.classifier._needs_specialized("Airport layout diagram"), "airport")
        self.assertIsNone(self.classifier._needs_specialized("Random text"))

    def test_classify_with_specialized(self):
        """Prueba classify cuando se requiere clasificación especializada."""
        with patch.object(self.classifier, '_clip_classification') as mock_clip, \
             patch.object(self.classifier, '_needs_specialized', return_value="chart"), \
             patch.object(self.classifier, '_specialized_classification') as mock_specialized:
            mock_clip.return_value = {"class": "Aeronautical navigation chart"}
            mock_specialized.return_value = {"class": "detailed_class"}
            result = self.classifier.classify("fake_image.png")
            self.assertIn("base_classification", result)
            self.assertIn("detailed_classification", result)


    def test_format_detections_empty(self):
        """Prueba el formateo cuando no hay detecciones."""
        result = _format_detections([])
        self.assertEqual(result, [], "Debería devolver lista vacía si no hay resultados")


if __name__ == '__main__':
    unittest.main()