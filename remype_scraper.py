import os
import sys
import time
import json
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import pytesseract
import numpy as np
import cv2
from captcha_analyzer import CaptchaAnalyzer

class RemypeScraper:
    def __init__(self, headless=False):
        self.base_url = "https://apps.trabajo.gob.pe/consultas-remype/app/index.html"
        self.headless = headless
        self.driver = None
        self.wait = None
        self.advanced_analyzer = CaptchaAnalyzer()
        
    def setup_driver(self):
        """Configurar driver básico sin bloqueos"""
        try:
            # Configurando Chrome WebDriver
            chrome_options = Options()
            
            # Directorio único para evitar conflictos
            import tempfile
            temp_dir = tempfile.mkdtemp()
            chrome_options.add_argument(f'--user-data-dir={temp_dir}')
            
            # Configuración mínima para evitar bloqueos
            if self.headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1280,720')
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--allow-running-insecure-content')
            
            # Evitar problemas de mutex y procesos
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-plugins')
            chrome_options.add_argument('--disable-background-networking')
            chrome_options.add_argument('--disable-background-timer-throttling')
            chrome_options.add_argument('--disable-renderer-backgrounding')
            chrome_options.add_argument('--disable-backgrounding-occluded-windows')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, 15)
            return True
        except Exception as e:
            print(f"❌ Error configurando driver: {e}")
            return False
            
    def close_modals(self):
        """Cerrar modales y popups iniciales"""
        try:
            time.sleep(3)
            
            # Selectores comunes para cerrar modales
            close_selectors = [
                "button[class*='close']",
                "button[aria-label='Close']",
                ".modal-close",
                ".close",
                "[data-dismiss='modal']",
                "button:contains('×')",
                "button:contains('Cerrar')",
                "button:contains('OK')",
                ".swal2-close",
                ".swal2-confirm"
            ]
            
            for selector in close_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed():
                            self.driver.execute_script("arguments[0].click();", element)
                            time.sleep(1)
                            break
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
            
    def find_ruc_input(self):
        """Encontrar campo de RUC"""
        selectors = [
            "input[name='ruc']",
            "input[id='ruc']",
            "input[placeholder*='RUC']",
            "input[type='text']"
        ]
        
        for selector in selectors:
            try:
                element = WebDriverWait(self.driver, 0.5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                if element.is_displayed():
                    return element
            except TimeoutException:
                continue
            except Exception as e:
                continue
        
        return None
        
    def get_captcha_image(self):
        """Obtener imagen de captcha con múltiples selectores"""
        try:
            captcha_selectors = [
                "img[src*='captcha']",
                "img[alt*='captcha']",
                "img[id*='captcha']",
                ".captcha img",
                "img[src*='Captcha']",
                "img[src*='CAPTCHA']",
                "canvas",
                "img[title*='captcha']",
                "img[class*='captcha']",
                "#captcha img",
                ".verification img",
                "img[src*='verification']"
            ]
            
            for i, selector in enumerate(captcha_selectors):
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for element in elements:
                        if element.is_displayed() and element.size['width'] > 50 and element.size['height'] > 20:
                            # Guardar imagen
                            timestamp = int(time.time())
                            captcha_path = f"captcha_{timestamp}.png"
                            element.screenshot(captcha_path)
                            
                            # Verificar que la imagen se guardó correctamente
                            if os.path.exists(captcha_path) and os.path.getsize(captcha_path) > 1000:
                                return captcha_path
                            else:
                                if os.path.exists(captcha_path):
                                    os.remove(captcha_path)
                except Exception as e:
                    continue
            
            return None
        except Exception as e:
            return None
            
    def preprocess_image_advanced(self, image):
        """Preprocesamiento avanzado de imagen para mejorar OCR"""
        processed_images = []
        
        try:
            # Convertir PIL a numpy array para procesamiento OpenCV
            img_array = np.array(image)
            
            # 1. Imagen original con mejoras básicas
            img1 = image.copy()
            enhancer = ImageEnhance.Contrast(img1)
            img1 = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(img1)
            img1 = enhancer.enhance(1.2)
            processed_images.append(("original_mejorada", img1))
            
            # 2. Filtro bilateral para reducir ruido manteniendo bordes
            img_bilateral = cv2.bilateralFilter(img_array, 9, 75, 75)
            img2 = Image.fromarray(img_bilateral)
            enhancer = ImageEnhance.Contrast(img2)
            img2 = enhancer.enhance(1.3)
            processed_images.append(("filtro_bilateral", img2))
            
            # 3. Umbralización OTSU (automática)
            img_blur = cv2.GaussianBlur(img_array, (3, 3), 0)
            _, img_otsu = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img3 = Image.fromarray(img_otsu)
            processed_images.append(("otsu_threshold", img3))
            
            # 4. Umbralización adaptativa mejorada
            img_adaptive = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
            img4 = Image.fromarray(img_adaptive)
            processed_images.append(("adaptativo_mejorado", img4))
            
            # 5. Eliminación de ruido con morfología específica para texto
            kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Aplicar closing para conectar letras fragmentadas
            img_morph1 = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel_rect)
            # Aplicar opening para eliminar ruido pequeño
            img_morph1 = cv2.morphologyEx(img_morph1, cv2.MORPH_OPEN, kernel_ellipse)
            img5 = Image.fromarray(img_morph1)
            processed_images.append(("morfologia_texto", img5))
            
            # 6. Detección de bordes + dilatación para resaltar caracteres
            edges = cv2.Canny(img_array, 50, 150)
            kernel_dilate = np.ones((2,2), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=1)
            # Invertir para que el texto sea blanco
            edges_inverted = cv2.bitwise_not(edges_dilated)
            img6 = Image.fromarray(edges_inverted)
            processed_images.append(("bordes_dilatados", img6))
            
            # 7. Filtro de mediana + contraste alto
            img_median = cv2.medianBlur(img_array, 3)
            img7 = Image.fromarray(img_median)
            enhancer = ImageEnhance.Contrast(img7)
            img7 = enhancer.enhance(2.5)
            processed_images.append(("mediana_contraste", img7))
            
            # 8. Imagen invertida con procesamiento
            img_inverted = cv2.bitwise_not(img_array)
            img_inv_blur = cv2.GaussianBlur(img_inverted, (3, 3), 0)
            _, img_inv_thresh = cv2.threshold(img_inv_blur, 127, 255, cv2.THRESH_BINARY)
            img8 = Image.fromarray(img_inv_thresh)
            processed_images.append(("invertida_procesada", img8))
            
            # 9. Combinación de técnicas para captchas específicos
            # Aplicar filtro bilateral + OTSU + morfología
            img_combo = cv2.bilateralFilter(img_array, 5, 50, 50)
            _, img_combo = cv2.threshold(img_combo, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel_combo = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            img_combo = cv2.morphologyEx(img_combo, cv2.MORPH_CLOSE, kernel_combo)
            img9 = Image.fromarray(img_combo)
            processed_images.append(("combo_captcha", img9))
            
        except Exception as e:
            print(f"⚠️ Error en procesamiento OpenCV: {e}")
            # Fallback a procesamiento básico con PIL
            img_basic = image.copy()
            enhancer = ImageEnhance.Contrast(img_basic)
            img_basic = enhancer.enhance(2.0)
            processed_images.append(("basico_fallback", img_basic))
        
        return processed_images
    
    def solve_captcha_basic(self, image_path):
        """Resolver captcha usando el analizador avanzado"""
        pass
        
        try:
            # Usar el analizador avanzado
            result = self.advanced_analyzer.analyze_captcha_comprehensive(image_path)
            
            if result and 'best_result' in result and result['best_result']:
                best = result['best_result']
                captcha_text = best.get('text', '')
                
                if captcha_text:
                    print(f"Captcha descifrado: {captcha_text}")
                    return captcha_text
                else:
                    print("El analizador avanzado no pudo extraer texto válido")
                    return None
            else:
                print("El analizador avanzado no encontró resultados")
                return None
                
        except Exception as e:
            print(f"Error en análisis avanzado: {str(e)}")
            print("Intentando con método básico como respaldo...")
            
            # Método de respaldo básico
            try:
                original_img = cv2.imread(image_path)
                if original_img is None:
                    return None
                
                # Procesamiento básico
                gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
                # OCR básico
                config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                text = pytesseract.image_to_string(resized, config=config).strip()
                text = ''.join(c for c in text if c.isalnum())
                
                if len(text) >= 3 and len(text) <= 8:
                    return text
                else:
                    print("Método de respaldo también falló")
                    return None
                    
            except Exception as backup_error:
                print(f"Error en método de respaldo: {str(backup_error)}")
                return None
    
    def correct_common_ocr_errors(self, text):
        """Corregir errores comunes de OCR en captchas"""
        # Diccionario de correcciones comunes
        corrections = {
            '0': 'O',  # Cero por O
            'l': '1',  # l minúscula por 1
            'I': '1',  # I mayúscula por 1
            '5': 'S',  # 5 por S
            '8': 'B',  # 8 por B
            '6': 'G',  # 6 por G
            '2': 'Z',  # 2 por Z
            '|': '1',  # Barra vertical por 1
        }
        
        corrected_text = text
        for wrong, correct in corrections.items():
            corrected_text = corrected_text.replace(wrong, correct)
        
        return corrected_text
    
    def validate_captcha_format(self, text):
        """Validar si el texto tiene formato típico de captcha REMYPE"""
        if not text or len(text) < 3:
            return False
        
        # Los captchas REMYPE suelen tener 4-6 caracteres alfanuméricos
        if not (3 <= len(text) <= 8):
            return False
        
        # Debe contener solo caracteres alfanuméricos
        if not text.isalnum():
            return False
        
        # No debe tener más de 2 caracteres iguales consecutivos
        for i in range(len(text) - 2):
            if text[i] == text[i+1] == text[i+2]:
                return False
        
        return True
    
    def calculate_confidence(self, text, img_type):
        """Calcular confianza basada en múltiples factores mejorados"""
        confidence = 0
        
        # Validación básica de formato
        if not self.validate_captcha_format(text):
            return 0
        
        # Factor de longitud optimizado para captchas REMYPE
        if len(text) == 5:  # Longitud más común
            confidence += 40
        elif len(text) == 4 or len(text) == 6:
            confidence += 35
        elif len(text) == 3:
            confidence += 25
        else:
            confidence += 15
        
        # Factor de caracteres válidos
        if text.isalnum():
            confidence += 20
        
        # Análisis de composición de caracteres
        letters = sum(1 for c in text if c.isalpha())
        numbers = sum(1 for c in text if c.isdigit())
        
        # Captchas REMYPE suelen tener mezcla de letras y números
        if letters > 0 and numbers > 0:
            confidence += 25
        elif letters > 0 or numbers > 0:
            confidence += 15
        
        # Bonus por diversidad de caracteres
        unique_chars = len(set(text))
        if unique_chars == len(text):  # Todos diferentes
            confidence += 20
        elif unique_chars >= len(text) * 0.8:  # Mayoría diferentes
            confidence += 15
        elif unique_chars >= len(text) * 0.6:
            confidence += 10
        else:
            confidence -= 15  # Penalizar repetición excesiva
        
        # Bonus por tipo de procesamiento de imagen
        processing_bonus = {
            "otsu_threshold": 25,
            "combo_captcha": 20,
            "morfologia_texto": 18,
            "adaptativo_mejorado": 15,
            "filtro_bilateral": 12,
            "bordes_dilatados": 10,
            "mediana_contraste": 8,
            "original_mejorada": 5,
            "invertida_procesada": 5
        }
        confidence += processing_bonus.get(img_type, 0)
        
        # Penalizar patrones sospechosos
        # Demasiadas letras consecutivas iguales
        for i in range(len(text) - 1):
            if text[i] == text[i+1]:
                confidence -= 5
        
        # Penalizar caracteres que raramente aparecen en captchas
        rare_chars = set('QXZ')
        for char in text:
            if char.upper() in rare_chars:
                confidence -= 3
        
        # Bonus por patrones comunes en captchas
        if any(c.isupper() for c in text) and any(c.islower() for c in text):
            confidence += 10  # Mezcla de mayúsculas y minúsculas
        
        return max(0, min(100, confidence))  # Limitar entre 0 y 100
            
    def find_captcha_input(self):
        """Encontrar campo de captcha"""
        selectors = [
            "input[name*='captcha']",
            "input[id*='captcha']",
            "input[placeholder*='código']",
            "input[type='text']:not([name='ruc'])"
        ]
        
        for selector in selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                if element.is_displayed():
                    return element
            except:
                continue
        return None
        
    def find_search_button(self):
        """Encontrar botón de búsqueda"""
        selectors = [
            ".btn-red",  # Selector específico encontrado en el HTML
            "button[data-ng-click*='buscar']",  # Selector específico del botón Angular
            "input[type='submit']",
            "button[type='submit']",
            "input[value*='Consultar']",
            "button:contains('Consultar')",
            ".btn-primary",
            "button[value*='Buscar']",
            "input[value*='Buscar']",
            "button:contains('Buscar')",
            "i.fa-search",
            ".fa-search",
            "[class*='search']",
            "[title*='Buscar']",
            "[title*='buscar']",
            "button .glyphicon-search",
            ".glyphicon-search",
            "button[onclick*='buscar']",
            "button[onclick*='Buscar']",
            "a[onclick*='buscar']",
            "a[onclick*='Buscar']"
        ]
        
        for selector in selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                if element.is_displayed():
                    return element
            except:
                continue
        return None
        
    def extract_table_data(self):
        """Extraer datos de la tabla REMYPE"""
        try:
            # Esperar a que carguen los datos
            time.sleep(3)
            
            data = []
            
            # Buscar tablas en la página
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            
            for table in tables:
                try:
                    # Buscar filas de datos (tr)
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    
                    for row in rows:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        
                        # Si la fila tiene datos (no es header)
                        if len(cells) >= 3:
                            row_data = {}
                            
                            # Extraer texto de cada celda
                            cell_texts = [cell.text.strip() for cell in cells]
                            
                            # Verificar si es una fila de datos válida (contiene RUC)
                            if any(text.isdigit() and len(text) >= 8 for text in cell_texts):
                                # Mapear datos según la estructura típica de REMYPE
                                if len(cell_texts) >= 7:
                                    row_data = {
                                        'RUC': cell_texts[0] if cell_texts[0] else '',
                                        'Razón Social': cell_texts[1] if len(cell_texts) > 1 else '',
                                        'Fecha Solicitud': cell_texts[2] if len(cell_texts) > 2 else '',
                                        'Estado/Condición': cell_texts[3] if len(cell_texts) > 3 else '',
                                        'Fecha Acreditación': cell_texts[4] if len(cell_texts) > 4 else '',
                                        'Situación Actual': cell_texts[5] if len(cell_texts) > 5 else '',
                                        'Documento': cell_texts[6] if len(cell_texts) > 6 else '',
                                        'Fecha Consulta': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    }
                                elif len(cell_texts) >= 4:
                                    row_data = {
                                        'RUC': cell_texts[0] if cell_texts[0] else '',
                                        'Razón Social': cell_texts[1] if len(cell_texts) > 1 else '',
                                        'Estado': cell_texts[2] if len(cell_texts) > 2 else '',
                                        'Fecha': cell_texts[3] if len(cell_texts) > 3 else '',
                                        'Fecha Consulta': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    }
                                
                                if row_data and row_data.get('RUC'):
                                    data.append(row_data)
                                    
                except Exception as e:
                    continue
            
            return data
            
        except Exception as e:
            return []
    
    def save_to_excel(self, data, ruc):
        """Guardar datos en archivo Excel"""
        try:
            if not data:
                # Si no hay datos, crear registro indicando que no se encontraron resultados
                data = [{
                    'RUC': ruc,
                    'Razón Social': 'NO ENCONTRADO',
                    'Estado': 'NO ES MYPE',
                    'Fecha Consulta': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Observaciones': 'No se encontraron resultados para esta búsqueda'
                }]
            
            # Crear DataFrame
            df = pd.DataFrame(data)
            
            # Nombre del archivo con timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'consulta_remype_{ruc}_{timestamp}.xlsx'
            
            # Guardar en Excel
            df.to_excel(filename, index=False, engine='openpyxl')
            
            print(f"📊 Datos guardados en: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error guardando Excel: {e}")
            return None
    
    def analyze_results(self):
        """Analizar resultados de la búsqueda"""
        try:
            # Esperar a que carguen los datos
            time.sleep(2)
            
            page_source = self.driver.page_source.lower()
            page_text = page_source.replace('\n', ' ').replace('\t', ' ')
            
            # Indicadores de error de captcha
            error_indicators = [
                "captcha incorrecto",
                "código incorrecto",
                "error en el captcha",
                "error en validar el captcha",
                "captcha inválido",
                "código de verificación incorrecto",
                "ingrese el código correcto",
                "el código ingresado no es válido",
                "captcha no válido",
                "verificación fallida"
            ]
            
            for indicator in error_indicators:
                if indicator in page_text:
                    return "error_captcha"
                    
            # PRIMERO: Buscar indicadores positivos de MYPE (prioridad alta)
            mype_indicators = [
                "acreditado como micro empresa",
                "acreditado como pequeña empresa",
                "vigente",
                "activo"
            ]
            
            # Verificar si hay indicadores positivos de MYPE
            for indicator in mype_indicators:
                if indicator in page_text:
                    # Verificar que no sea solo parte de un encabezado
                    context_start = max(0, page_text.find(indicator) - 100)
                    context_end = min(len(page_text), page_text.find(indicator) + 100)
                    context = page_text[context_start:context_end]
                    
                    # Si no está en un encabezado (bgcolor), es un dato real
                    if "bgcolor=\"#666666\"" not in context:
                        return "es_mype"
            
            # SEGUNDO: Buscar datos específicos en las tablas
            table_data_indicators = [
                "fecha de acreditación",
                "situación actual",
                "estado/condición"
            ]
            
            has_real_data = False
            for indicator in table_data_indicators:
                if indicator in page_text:
                    # Verificar que no sea solo el encabezado
                    context_start = max(0, page_text.find(indicator) - 100)
                    context_end = min(len(page_text), page_text.find(indicator) + 100)
                    context = page_text[context_start:context_end]
                    
                    if "bgcolor=\"#666666\"" not in context:
                        has_real_data = True
                        break
            
            if has_real_data:
                return "es_mype"
            
            # TERCERO: Solo entonces buscar indicadores de no MYPE
            no_mype_indicators = [
                "no se encontraron resultados para esta busqueda",
                "no se encontraron registros para esta busqueda"
            ]
            
            # Verificar que realmente no hay datos (no solo el mensaje genérico)
            no_data_found = True
            for indicator in no_mype_indicators:
                if indicator in page_text:
                    # Verificar que no haya datos reales en otra parte
                    if "acreditado" not in page_text or "micro empresa" not in page_text:
                        return "no_es_mype"
                    else:
                        no_data_found = False
                        break
                    
            return "indeterminado"
            
        except Exception as e:
            return None
            
    def scrape_ruc(self, ruc, max_attempts=5):
        """Función principal de scraping"""
        try:
            # Iniciando consulta
            
            if not self.setup_driver():
                return None
                
            # Navegar a la página
            self.driver.get(self.base_url)
            time.sleep(3)
            
            # Cerrar modales
            self.close_modals()
            
            for attempt in range(max_attempts):
                try:
                    # Encontrar campo RUC
                    ruc_input = self.find_ruc_input()
                    if not ruc_input:
                        continue
                        
                    # Ingresar RUC
                    ruc_input.clear()
                    ruc_input.send_keys(ruc)
                    
                    # Obtener captcha
                    captcha_path = self.get_captcha_image()
                    if not captcha_path:
                        continue
                        
                    # Resolver captcha
                    captcha_solution = self.solve_captcha_basic(captcha_path)
                    if not captcha_solution:
                        continue
                        
                    # Ingresar captcha
                    captcha_input = self.find_captcha_input()
                    if not captcha_input:
                        continue
                        
                    captcha_input.clear()
                    time.sleep(1)  # Pausa antes de ingresar
                    captcha_input.send_keys(captcha_solution)
                    
                    # Esperar un momento para que se procese
                    time.sleep(3)
                    
                    # Capturar HTML para análisis del botón de búsqueda
                    page_html = self.driver.page_source
                    
                    # Hacer clic en buscar
                    search_button = self.find_search_button()
                    if not search_button:
                        continue
                        
                    search_button.click()
                    
                    # Esperar más tiempo para que cargue la página de resultados
                    time.sleep(8)  # Aumentar tiempo de espera
                    
                    # Verificar si hay errores de captcha antes de analizar
                    page_source = self.driver.page_source.lower()
                    
                    # Lista ampliada de indicadores de error de captcha
                    captcha_error_indicators = [
                        "captcha incorrecto",
                        "código incorrecto",
                        "error en validar el captcha",
                        "captcha inválido",
                        "código de verificación incorrecto",
                        "ingrese el código correcto",
                        "el código ingresado no es válido"
                    ]
                    
                    captcha_error_found = False
                    for indicator in captcha_error_indicators:
                        if indicator in page_source:
                            captcha_error_found = True
                            break
                    
                    if captcha_error_found:
                        # Recargar la página para obtener un nuevo captcha
                        self.driver.refresh()
                        time.sleep(3)
                        self.close_modals()
                        continue
                    
                    # Analizar resultados
                    result = self.analyze_results()
                    
                    # Esperar un poco más antes de procesar el resultado
                    time.sleep(3)
                    
                    if result == "error_captcha":
                        # Recargar la página para obtener un nuevo captcha
                        self.driver.refresh()
                        time.sleep(3)
                        self.close_modals()
                        continue
                    elif result in ["es_mype", "no_es_mype"]:
                        # Extraer datos de la tabla y guardar en Excel
                        table_data = self.extract_table_data()
                        excel_file = self.save_to_excel(table_data, ruc)
                        return result
                    elif result == "indeterminado":
                        # Recargar la página para intentar de nuevo
                        self.driver.refresh()
                        time.sleep(3)
                        self.close_modals()
                        continue
                        
                except Exception as e:
                    continue
                    
            return None
            
        except Exception as e:
            return None
        finally:
            if self.driver:
                self.driver.quit()
                
    def cleanup_temp_files(self):
        """Limpiar archivos temporales"""
        try:
            for file in os.listdir('.'):
                if file.startswith('captcha_') and file.endswith('.png'):
                    os.remove(file)
        except Exception as e:
            print(f"⚠️ Error limpiando archivos: {e}")

def main():
    """Función principal"""
    if len(sys.argv) != 2:
        print("Uso: python remype_scraper_simple.py <RUC>")
        print("Ejemplo: python remype_scraper_simple.py 20513345748")
        sys.exit(1)
        
    ruc = sys.argv[1].strip()
    
    if not ruc.isdigit() or len(ruc) != 11:
        print("Error: El RUC debe tener exactamente 11 dígitos")
        sys.exit(1)
        
    # Inicializando consulta REMYPE
    
    scraper = RemypeScraper(headless=False)
    result = scraper.scrape_ruc(ruc)
    
    if result == "es_mype":
        print("La empresa SÍ es MYPE")
    elif result == "no_es_mype":
        print("La empresa NO es MYPE")
    else:
        print("No se pudo determinar")
        
    # Limpiar archivos temporales
    scraper.cleanup_temp_files()

if __name__ == "__main__":
    main()