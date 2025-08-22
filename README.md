# Scraper REMYPE - Consulta Automática de MYPE

🔍 **Sistema automatizado para consultar el estado MYPE de empresas en el Registro Nacional de la Micro y Pequeña Empresa (REMYPE) del Perú.**

## 📋 Características Principales

- ✅ **Consulta automática** de RUCs en el sistema REMYPE
- 🤖 **Resolución inteligente de captcha** con análisis de imagen avanzado
- 📊 **Generación automática de Excel** con los datos extraídos
- 🚀 **Interfaz optimizada** con mensajes mínimos y ejecución rápida
- 🔄 **Sistema de reintentos** automático en caso de errores
- 📱 **Detección precisa** del estado MYPE de las empresas

## 🛠️ Tecnologías Utilizadas

- **Python 3.7+**
- **Selenium WebDriver** - Automatización web
- **OpenCV** - Procesamiento de imágenes
- **Tesseract OCR** - Reconocimiento óptico de caracteres
- **Pandas** - Manipulación de datos
- **OpenPyXL** - Generación de archivos Excel

## 📦 Instalación

### Prerrequisitos

1. **Python 3.7 o superior**
2. **Google Chrome** instalado
3. **Tesseract OCR** instalado:
   ```bash
   # macOS
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # Windows
   # Descargar desde: https://github.com/UB-Mannheim/tesseract/wiki
   ```

### Instalación de dependencias

```bash
# Clonar o descargar el proyecto
cd Web-Scrapping

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## 🚀 Uso

### Consulta básica

```bash
python3 remype_scraper_simple.py [RUC]
```

**Ejemplo:**
```bash
python3 remype_scraper_simple.py 20513345748
```

### Salida esperada

```
Captcha descifrado: zUBey
📊 Datos guardados en: consulta_remype_20513345748_20250821_212046.xlsx
✅ La empresa SÍ es MYPE
```

## 📊 Archivo Excel Generado

Cada consulta genera automáticamente un archivo Excel con:

| Campo | Descripción |
|-------|-------------|
| **RUC** | Número de RUC consultado |
| **Razón Social** | Nombre de la empresa |
| **Fecha Solicitud** | Fecha de solicitud MYPE |
| **Estado/Condición** | Estado actual (ej: "ACREDITADO COMO MICRO EMPRESA") |
| **Fecha Acreditación** | Fecha de acreditación |
| **Situación Actual** | Situación vigente |
| **Documento** | Documento de sustento |
| **Fecha Consulta** | Timestamp de la consulta |

## 🏗️ Estructura del Proyecto

```
Web-Scrapping/
├── remype_scraper_simple.py      # Scraper principal
├── captcha_analyzer_simple.py    # Analizador de captcha
├── requirements.txt              # Dependencias Python
├── README.md                     # Documentación
└── consulta_remype_*.xlsx        # Archivos Excel generados
```

## ⚙️ Configuración Avanzada

### Modo headless

Para ejecutar sin interfaz gráfica, modificar en `remype_scraper_simple.py`:

```python
scraper = RemypeScraperSimple(headless=True)
```

### Timeout personalizado

Ajustar tiempos de espera en la clase `RemypeScraperSimple`:

```python
WebDriverWait(self.driver, timeout_seconds)
```

## 🔧 Funcionalidades Técnicas

### Sistema de Captcha
- **Múltiples técnicas de procesamiento** de imagen
- **Análisis morfológico** y filtros adaptativos
- **OCR con configuraciones optimizadas**
- **Sistema de puntuación** para seleccionar la mejor solución

### Extracción de Datos
- **Detección automática** de tablas HTML
- **Mapeo inteligente** de columnas
- **Validación de datos** extraídos
- **Manejo de diferentes formatos** de tabla

### Generación de Reportes
- **Archivos Excel** con formato profesional
- **Timestamps únicos** para evitar sobrescritura
- **Manejo de casos sin resultados**
- **Metadatos de consulta** incluidos

## 🐛 Solución de Problemas

### Error: "No module named 'openpyxl'"
```bash
pip install openpyxl pandas
```

### Error: "Tesseract not found"
- Verificar instalación de Tesseract OCR
- Agregar Tesseract al PATH del sistema

### Error: "ChromeDriver not found"
- El script descarga automáticamente ChromeDriver
- Verificar que Google Chrome esté instalado

### Captcha no se resuelve
- El sistema reintenta automáticamente hasta 5 veces
- Verificar conexión a internet
- Comprobar que el sitio REMYPE esté disponible

## 📈 Resultados Posibles

- ✅ **"La empresa SÍ es MYPE"** - Empresa registrada como MYPE
- ❌ **"La empresa NO es MYPE"** - Empresa no registrada o sin resultados
- ⚠️ **"No se pudo determinar"** - Error en la consulta

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto es de uso educativo y de investigación. Respetar los términos de uso del sitio web REMYPE.

## ⚠️ Disclaimer

Este software es para fines educativos y de automatización personal. El usuario es responsable de cumplir con los términos de servicio del sitio web REMYPE y las leyes aplicables.

---

**Desarrollado con ❤️ para automatizar consultas REMYPE**