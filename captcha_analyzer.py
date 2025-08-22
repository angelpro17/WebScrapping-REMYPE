import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import pytesseract
import os
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans

class CaptchaAnalyzer:
    def __init__(self):
        self.debug_mode = True
        self.save_debug_images = False
        self.debug_dir = "captcha_debug"
        
        # Crear directorio de debug si no existe
        if self.save_debug_images and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
    
    def analyze_captcha_comprehensive(self, image_path: str) -> Dict:
        """Análisis completo del captcha"""
        # Análisis silencioso
        
        # Cargar imagen original
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"No se pudo cargar la imagen: {image_path}")
            return {}
        
        # Análisis de propiedades básicas
        analysis = self.analyze_image_properties(original_img)
        
        # Generar todas las versiones procesadas
        processed_images = self.generate_all_processed_versions(original_img)
        
        # Análisis OCR en todas las versiones
        ocr_results = self.comprehensive_ocr_analysis(processed_images)
        
        # Seleccionar mejor resultado
        best_result = self.select_best_result(ocr_results)
        
        return {
            'best_result': best_result,
            'all_results': ocr_results,
            'image_analysis': analysis,
            'processed_versions': len(processed_images)
        }
    
    def analyze_image_properties(self, img: np.ndarray) -> Dict:
        """Analizar propiedades básicas de la imagen"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Estadísticas básicas
        height, width = gray.shape
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Análisis de contraste
        contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
        
        # Detección de bordes para medir complejidad
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Análisis de histograma
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_peaks = len([i for i in range(1, 255) if hist[i] > hist[i-1] and hist[i] > hist[i+1]])
        
        return {
            'dimensions': (width, height),
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'contrast_ratio': contrast,
            'edge_density': edge_density,
            'histogram_peaks': hist_peaks,
            'is_low_contrast': contrast < 0.3,
            'is_noisy': edge_density > 0.1
        }
    
    def generate_all_processed_versions(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Generar todas las versiones procesadas de la imagen"""
        processed = {}
        
        # 1. Imagen original en escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed['original_gray'] = gray
        
        # 2. Redimensionado con interpolación cúbica
        height, width = gray.shape
        new_width = int(width * 2.5)
        new_height = int(height * 2.5)
        resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        processed['resized_cubic'] = resized
        
        # 3. Mejora de contraste adaptativa (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gray)
        processed['clahe_enhanced'] = clahe_img
        
        # 4. Filtro bilateral para reducir ruido manteniendo bordes
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        processed['bilateral_filtered'] = bilateral
        
        # 5. Umbralización OTSU
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed['otsu_threshold'] = otsu
        
        # 6. Umbralización adaptativa Gaussiana
        adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed['adaptive_gaussian'] = adaptive_gaussian
        
        # 7. Umbralización adaptativa media
        adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        processed['adaptive_mean'] = adaptive_mean
        
        # 8. Operaciones morfológicas para limpiar texto
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_open = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
        morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)
        processed['morphological_cleaned'] = morph_close
        
        # 9. Detección de bordes Canny con dilatación
        edges = cv2.Canny(gray, 50, 150)
        kernel_dilate = np.ones((2,2), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel_dilate, iterations=1)
        processed['canny_dilated'] = dilated_edges
        
        # 10. Filtro de mediana para reducir ruido sal y pimienta
        median_filtered = cv2.medianBlur(gray, 3)
        processed['median_filtered'] = median_filtered
        
        # 11. Sharpening con kernel personalizado
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        processed['sharpened'] = sharpened
        
        # 12. Imagen invertida
        inverted = cv2.bitwise_not(gray)
        processed['inverted'] = inverted
        
        # 13. Combinación de técnicas para captchas específicos
        # Aplicar CLAHE + bilateral + OTSU
        combined = clahe.apply(bilateral)
        _, combined_thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed['combined_advanced'] = combined_thresh
        
        # 14. Técnica de segmentación por clustering de colores (simplificada)
        if len(img.shape) == 3:
            try:
                clustered = self.color_clustering_segmentation(img)
                processed['color_clustered'] = clustered
            except:
                pass  # Si falla, continuar sin esta técnica
        
        return processed
    
    def color_clustering_segmentation(self, img: np.ndarray, k: int = 3) -> np.ndarray:
        """Segmentación por clustering de colores"""
        data = img.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convertir de vuelta a uint8 y reshape
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_img = segmented_data.reshape(img.shape)
        
        # Convertir a escala de grises
        return cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    
    def comprehensive_ocr_analysis(self, processed_images: Dict[str, np.ndarray]) -> List[Dict]:
        """Análisis OCR completo en todas las versiones procesadas"""
        all_results = []
        
        # Configuraciones OCR avanzadas
        ocr_configs = [
            '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
            '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
            '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
            '--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
            '--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
            '--oem 2 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
        ]
        
        for img_name, img_array in processed_images.items():
            # Procesando imagen silenciosamente
            
            for i, config in enumerate(ocr_configs):
                try:
                    # OCR con datos de confianza
                    data = pytesseract.image_to_data(img_array, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Extraer texto y confianza
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    texts = [text.strip() for text in data['text'] if text.strip()]
                    
                    if texts and confidences:
                        combined_text = ''.join(texts)
                        avg_confidence = sum(confidences) / len(confidences)
                        
                        # Filtrar solo caracteres alfanuméricos
                        clean_text = ''.join(c for c in combined_text if c.isalnum())
                        
                        if len(clean_text) >= 3 and len(clean_text) <= 8:
                            result = {
                                'text': clean_text,
                                'confidence': avg_confidence,
                                'image_type': img_name,
                                'ocr_config': i,
                                'char_count': len(clean_text),
                                'has_numbers': any(c.isdigit() for c in clean_text),
                                'has_letters': any(c.isalpha() for c in clean_text),
                                'is_mixed': any(c.isdigit() for c in clean_text) and any(c.isalpha() for c in clean_text)
                            }
                            
                            # Calcular score personalizado
                            result['custom_score'] = self.calculate_advanced_confidence(result)
                            all_results.append(result)
                            
                except Exception as e:
                    continue
        
        return all_results
    
    def calculate_advanced_confidence(self, result: Dict) -> float:
        """Calcular confianza avanzada basada en múltiples factores"""
        score = result['confidence']
        
        # Bonus por longitud óptima (5-6 caracteres es común en captchas)
        if 5 <= result['char_count'] <= 6:
            score += 10
        elif result['char_count'] == 4:
            score += 5
        
        # Bonus por mezcla de números y letras
        if result['is_mixed']:
            score += 15
        
        # Bonus por tipos de imagen específicos
        img_bonuses = {
            'clahe_enhanced': 10,
            'combined_advanced': 15,
            'bilateral_filtered': 8,
            'otsu_threshold': 12,
            'morphological_cleaned': 10
        }
        
        if result['image_type'] in img_bonuses:
            score += img_bonuses[result['image_type']]
        
        # Penalización por texto muy corto o muy largo
        if result['char_count'] < 3 or result['char_count'] > 8:
            score -= 20
        
        return max(0, score)
    
    def select_best_result(self, results: List[Dict]) -> Dict:
        """Seleccionar el mejor resultado basado en score personalizado"""
        if not results:
            return {}
        
        # Ordenar por score personalizado
        sorted_results = sorted(results, key=lambda x: x['custom_score'], reverse=True)
        
        # Top 5 resultados calculados silenciosamente
        for i, result in enumerate(sorted_results[:5]):
            pass
        
        return sorted_results[0] if sorted_results else {}

# Función de utilidad para usar el analizador
def analyze_captcha_file(image_path: str) -> str:
    """Función simple para analizar un archivo de captcha"""
    analyzer = CaptchaAnalyzer()
    result = analyzer.analyze_captcha_comprehensive(image_path)
    
    if result and 'best_result' in result and result['best_result']:
        return result['best_result']['text']
    return ""

if __name__ == "__main__":
    # Ejemplo de uso
    analyzer = SimpleCaptchaAnalyzer()
    
    # Buscar archivos de captcha en el directorio actual
    captcha_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'captcha' in f.lower()]
    
    if captcha_files:
        print(f"🔍 Encontrados {len(captcha_files)} archivos de captcha")
        for captcha_file in captcha_files[:3]:  # Analizar solo los primeros 3
            print(f"\n{'='*50}")
            print(f"Analizando: {captcha_file}")
            print(f"{'='*50}")
            result = analyzer.analyze_captcha_comprehensive(captcha_file)
            if result and 'best_result' in result:
                print(f"\nResultado final: '{result['best_result'].get('text', 'N/A')}'")
    else:
        print("No se encontraron archivos de captcha en el directorio actual")
        print("Coloca archivos de captcha (.png, .jpg) con 'captcha' en el nombre")