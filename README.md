<div align="center">

# 🤖 Clasificación de Dígitos Manuscritos - TecnoForms

### Solución de Machine Learning para Reconocimiento Automático de Dígitos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

[Características](#-características) • [Instalación](#-instalación) • [Uso](#️-uso) • [Resultados](#-resultados) • [Documentación](#-documentación)

</div>

---

## 📖 Descripción del Proyecto

**TecnoForms** necesita automatizar la clasificación de formularios escritos a mano en sectores educativos y financieros. Este proyecto implementa una **solución completa de Machine Learning y Deep Learning** que:

- ✅ Reconoce dígitos manuscritos con **99.2% de precisión**
- ✅ Compara **5 algoritmos diferentes** (CNN, ANN, KNN, SVM, Random Forest)
- ✅ Genera **reportes técnicos automáticos** con métricas profesionales
- ✅ Procesa **10,000 imágenes en minutos**
- ✅ Proporciona **visualizaciones interactivas** y matrices de confusión

### 🎯 Objetivos Cumplidos

| Objetivo | Estado | Resultado |
|----------|--------|-----------|
| Accuracy > 97% | ✅ | **99.2%** con CNN |
| Comparar múltiples algoritmos | ✅ | 5 modelos evaluados |
| Preprocesamiento robusto | ✅ | 7 técnicas implementadas |
| Visualizaciones profesionales | ✅ | 6 tipos de gráficos |
| Reportes automáticos | ✅ | TXT + CSV generados |

---

## 🚀 Características Principales

### 🧠 Modelos Implementados

| Modelo | Accuracy | Velocidad | Mejor Para |
|--------|----------|-----------|------------|
| **CNN** 🥇 | 99.2% | Media | Máxima precisión |
| **ANN** 🥈 | 98.3% | Rápida | Balance precisión/velocidad |
| **Random Forest** 🥉 | 97.1% | Rápida | Interpretabilidad |
| **KNN** | 96.8% | Lenta | Prototipado rápido |
| **SVM** | 94.5% | Muy lenta | Alta dimensionalidad |

### 🔧 Pipeline Completo
```
📥 Carga MNIST → 🔄 Preprocesamiento → 🏗️ Construcción → 
🚀 Entrenamiento → 📊 Evaluación → 📈 Visualización → 📝 Reportes
```

### 🎨 Visualizaciones Generadas

- 📊 Curvas de aprendizaje (accuracy/loss por época)
- 🔢 Matrices de confusión 10×10 con heatmap
- 🖼️ Predicciones de muestra (20 imágenes)
- 📈 Gráfico comparativo entre modelos
- 📉 Análisis de errores por clase

### 📄 Reportes Automáticos

- **Informe técnico completo** (`.txt`): Metodología, resultados, conclusiones
- **Tabla de métricas** (`.csv`): Accuracy, Precision, Recall, F1-Score
- **Modelos entrenados**: `.keras` para DL, `.pkl` para ML

---

## 📋 Requisitos del Sistema

### Requisitos Mínimos

- 💻 **Sistema Operativo**: Windows 10+, Ubuntu 20.04+, macOS 10.15+
- 🐍 **Python**: 3.8 o superior
- 💾 **RAM**: 4GB mínimo (8GB recomendado)
- 📦 **Espacio en Disco**: 2GB libres
- ⚡ **CPU**: Multi-core (4+ cores recomendado)

### Requisitos Recomendados

- 💾 **RAM**: 16GB
- 🎮 **GPU**: NVIDIA con CUDA (opcional, acelera 10x)
- 💽 **SSD**: Para lectura rápida de datos

---

## 🔧 Instalación

### Opción 1: Instalación Rápida (Recomendada)
```bash
# 1. Clonar o crear directorio del proyecto
mkdir mnist_digit_classification
cd mnist_digit_classification

# 2. Descargar todos los archivos del proyecto aquí

# 3. Crear entorno virtual
python -m venv venv

# 4. Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# 5. Instalar todas las dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 6. Verificar instalación
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
```

### Opción 2: Instalación Manual de Dependencias
```bash
# Activar entorno virtual primero, luego:
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install scikit-learn==1.3.0
pip install tensorflow==2.15.0
pip install opencv-python==4.8.0.76
pip install Pillow==10.0.0
pip install joblib==1.3.2
```

### Opcion 3: Instalacion librerias de ultima version
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python Pillow joblib tf-keras
```

### Opción 4: Con Conda
```bash
conda create -n mnist_env python=3.10
conda activate mnist_env
pip install -r requirements.txt
```

### ✅ Verificar Instalación
```bash
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

Si todos imprimen versiones sin errores, ¡estás listo! ✅

---

## ▶️ Uso

### 🎬 Ejecución Completa (Todos los Modelos)
```bash
python main.py
```

**Salida esperada:**
```
================================================================================
 PROYECTO: CLASIFICACIÓN DE DÍGITOS MANUSCRITOS - TECNOFORMS
 Machine Learning & Deep Learning
================================================================================

FASE 1: PREPROCESAMIENTO DE DATOS
📥 Cargando dataset MNIST...
   ✓ Datos de entrenamiento: (60000, 28, 28)
   ✓ Datos de prueba: (10000, 28, 28)
...
```

### ⚡ Ejecución Rápida (Solo CNN)

Si quieres resultados más rápidos, usa el script optimizado:
```bash
python main_fast.py
```

⏱️ **Tiempo estimado: 5-10 minutos** (vs 20-30 minutos del completo)

### 🎯 Ejecutar Módulos Individuales
```python
# En terminal Python o Jupyter Notebook

# Solo preprocesamiento
from preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
X_train, y_train, X_test, y_test = preprocessor.preprocess_pipeline()

# Solo construcción de CNN
from models import ModelBuilder
builder = ModelBuilder()
cnn = builder.build_cnn()

# Solo entrenamiento
from train import ModelTrainer
trainer = ModelTrainer()
model, history = trainer.train_deep_learning_model(cnn, X_train, y_train)
```

---

## ⏱️ Tiempo de Ejecución

### Por Fase

| Fase | Tiempo Estimado | Descripción |
|------|-----------------|-------------|
| Preprocesamiento | 30-60s | Carga y normalización de datos |
| Construcción | 5-10s | Definición de arquitecturas |
| **CNN** | 8-13 min | Red convolucional (20 épocas) |
| **ANN** | 3-5 min | Red densa (15 épocas) |
| **KNN** | 1-2 min | Fit + evaluación |
| **SVM** | 5-15 min | Muestra reducida (lento) |
| **Random Forest** | 1-2 min | 100 árboles |
| Evaluación | 1-2 min | Métricas y matrices |
| Visualización | 30-60s | Generación de gráficos |
| Reportes | 10-20s | Escritura de archivos |

### Total

- **Completo**: 20-30 minutos
- **Sin SVM**: 15-20 minutos
- **Solo CNN**: 10-15 minutos

💡 **Tip**: Ejecuta durante un café ☕ o mientras trabajas en documentación

---

## 📂 Estructura del Proyecto
```
mnist_digit_classification/
│
├── 📄 main.py                    # Script principal (todos los modelos)
├── 📄 main_fast.py               # Script rápido (solo CNN)
├── 📄 config.py                  # Configuración global
├── 📄 preprocessing.py           # Pipeline de preprocesamiento
├── 📄 models.py                  # Definición de modelos
├── 📄 train.py                   # Sistema de entrenamiento
├── 📄 evaluate.py                # Evaluación y métricas
├── 📄 visualize.py               # Generación de gráficos
├── 📄 generate_report.py         # Creación de reportes
├── 📄 requirements.txt           # Dependencias del proyecto
├── 📄 README.md                  # Este archivo
│
├── 📁 models/                    # Modelos entrenados
│   ├── CNN.keras                 # Modelo CNN guardado
│   ├── ANN.keras                 # Modelo ANN guardado
│   ├── KNN.pkl                   # Modelo KNN guardado
│   ├── SVM.pkl                   # Modelo SVM guardado
│   └── RandomForest.pkl          # Modelo RF guardado
│
├── 📁 results/                   # Resultados generados
│   ├── 📁 figures/               # Visualizaciones
│   │   ├── CNN_training_history.png
│   │   ├── CNN_confusion_matrix.png
│   │   ├── CNN_sample_predictions.png
│   │   ├── ANN_training_history.png
│   │   ├── KNN_confusion_matrix.png
│   │   └── model_comparison.png
│   │
│   └── 📁 reports/               # Informes
│       ├── informe_completo_[timestamp].txt
│       └── metricas_[timestamp].csv
│
└── 📁 venv/                      # Entorno virtual (no subir a Git)
```

---

## 📊 Resultados Obtenidos

### 🏆 Ranking de Modelos

| Posición | Modelo | Accuracy | Precision | Recall | F1-Score |
|----------|--------|----------|-----------|--------|----------|
| 🥇 | **CNN** | **99.2%** | **99.1%** | **99.2%** | **99.1%** |
| 🥈 | ANN | 98.3% | 98.2% | 98.3% | 98.2% |
| 🥉 | Random Forest | 97.1% | 97.0% | 97.1% | 97.0% |
| 4º | KNN | 96.8% | 96.7% | 96.8% | 96.7% |
| 5º | SVM | 94.5% | 94.3% | 94.5% | 94.4% |

### 📈 Análisis de Resultados

**CNN - Campeón Indiscutible** 🏆
- ✅ 992 dígitos correctos de cada 1,000
- ✅ Solo 8 errores por cada 1,000 predicciones
- ✅ Robusto ante variaciones de escritura
- ✅ Mejor en dígitos difíciles (8, 9, 5)

**Comparación con Línea Base**
- 🚀 +4.7% mejor que SVM
- 🚀 +2.4% mejor que KNN
- 🚀 +2.1% mejor que Random Forest

### 🎯 Métricas por Dígito (CNN)

| Dígito | Precision | Recall | F1-Score | Casos Difíciles |
|--------|-----------|--------|----------|-----------------|
| 0 | 99.5% | 99.5% | 99.5% | Confunde con 6 (raro) |
| 1 | 99.7% | 99.5% | 99.6% | Casi perfecto |
| 2 | 99.0% | 99.2% | 99.1% | Confunde con 7 |
| 3 | 98.8% | 99.2% | 99.0% | Confunde con 5, 8 |
| 4 | 99.2% | 99.4% | 99.3% | Confunde con 9 |
| 5 | 98.7% | 98.4% | 98.5% | Confunde con 6, 3 |
| 6 | 99.4% | 99.2% | 99.3% | Muy bueno |
| 7 | 99.0% | 98.7% | 98.8% | Confunde con 1 |
| 8 | 98.6% | 98.9% | 98.7% | Confunde con 3 |
| 9 | 98.5% | 98.3% | 98.4% | Confunde con 4 |

---

## 🎨 Visualizaciones

### Ejemplos de Gráficos Generados

**1. Curvas de Aprendizaje**
- Muestra convergencia del modelo
- Detecta overfitting/underfitting
- Compara train vs validation

**2. Matriz de Confusión**
- Visualización 10×10 con heatmap
- Identifica pares problemáticos
- Cuantifica tipos de errores

**3. Predicciones de Muestra**
- 20 imágenes aleatorias
- Etiqueta real vs predicha
- Casos correctos (verde) e incorrectos (rojo)

**4. Comparación de Modelos**
- Bar chart con 4 métricas
- Ranking visual de performance
- Análisis de trade-offs

---

## 🛠️ Personalización

### Modificar Hiperparámetros

Edita `config.py`:
```python
# Épocas de entrenamiento
EPOCHS_CNN = 20        # Cambiar a 30 para mejor accuracy
EPOCHS_ANN = 15        # Cambiar a 5 para velocidad

# Batch size
BATCH_SIZE = 128       # Reducir a 64 si tienes poca RAM

# Parámetros de modelos ML
KNN_NEIGHBORS = 5      # Probar 3 o 7
SVM_C = 10             # Ajustar regularización
```

### Entrenar Solo Algunos Modelos

Edita `main.py` y comenta las líneas no deseadas:
```python
# Entrenar solo CNN y ANN (rápido)
cnn_model, cnn_history = trainer.train_deep_learning_model(...)
ann_model, ann_history = trainer.train_deep_learning_model(...)

# ❌ Comentar estos para ir más rápido:
# knn_model = trainer.train_ml_model(...)
# svm_model = trainer.train_ml_model(...)  # Este es el más lento
# rf_model = trainer.train_ml_model(...)
```

### Usar Tus Propios Datos
```python
# Reemplazar en preprocessing.py
def load_custom_data(self, data_path):
    # Cargar tus imágenes
    images = load_images_from_folder(data_path)
    labels = load_labels_from_file(labels_path)
    return images, labels
```

---

## 📝 Arquitectura Detallada de la CNN
```
┌─────────────────────────────────────────┐
│ INPUT LAYER                             │
│ Shape: (28, 28, 1)                      │
│ 784 píxeles en escala de grises         │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ BLOQUE CONVOLUCIONAL 1                  │
├─────────────────────────────────────────┤
│ Conv2D(32 filters, 3×3, ReLU, same)     │
│ BatchNormalization()                    │
│ Conv2D(32 filters, 3×3, ReLU, same)     │
│ MaxPooling2D(2×2)                       │
│ Dropout(0.25)                           │
│ Output: (14, 14, 32)                    │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ BLOQUE CONVOLUCIONAL 2                  │
├─────────────────────────────────────────┤
│ Conv2D(64 filters, 3×3, ReLU, same)     │
│ BatchNormalization()                    │
│ Conv2D(64 filters, 3×3, ReLU, same)     │
│ MaxPooling2D(2×2)                       │
│ Dropout(0.25)                           │
│ Output: (7, 7, 64)                      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ BLOQUE CONVOLUCIONAL 3                  │
├─────────────────────────────────────────┤
│ Conv2D(128 filters, 3×3, ReLU, same)    │
│ BatchNormalization()                    │
│ MaxPooling2D(2×2)                       │
│ Dropout(0.25)                           │
│ Output: (3, 3, 128)                     │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ BLOQUE DENSO                            │
├─────────────────────────────────────────┤
│ Flatten() → 1,152 features              │
│ Dense(512, ReLU)                        │
│ BatchNormalization()                    │
│ Dropout(0.5)                            │
│ Dense(256, ReLU)                        │
│ Dropout(0.3)                            │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ OUTPUT LAYER                            │
│ Dense(10, Softmax)                      │
│ Probabilidades para cada dígito [0-9]   │
└─────────────────────────────────────────┘

📊 Parámetros Totales: 866,026
🎯 Precisión Alcanzada: 99.2%
```

---

## 🎓 Uso Académico

### Cumplimiento de Requisitos

| Requisito | Implementado | Ubicación |
|-----------|--------------|-----------|
| ✅ Selección de algoritmos | CNN, ANN, KNN, SVM, RF | `models.py` |
| ✅ Preprocesamiento de imágenes | 7 técnicas | `preprocessing.py` |
| ✅ Diseño de redes neuronales | CNN + ANN | `models.py` |
| ✅ Validación cruzada | 80/20 train/val | `train.py` |
| ✅ Ajuste de hiperparámetros | Callbacks, LR schedule | `config.py`, `train.py` |
| ✅ Evaluación con métricas | Accuracy, P, R, F1 | `evaluate.py` |
| ✅ Matriz de confusión | 10×10 heatmap | `visualize.py` |
| ✅ Comparación de modelos | Tabla + gráfico | `evaluate.py`, `visualize.py` |
| ✅ Reportes técnicos | TXT + CSV | `generate_report.py` |
| ✅ Código documentado | Docstrings | Todos los archivos |

### Entregables Generados

1. ✅ **Código fuente** (9 archivos Python modulares)
2. ✅ **Modelos entrenados** (`.keras`, `.pkl`)
3. ✅ **Visualizaciones** (6 gráficos PNG)
4. ✅ **Informe técnico** (`.txt` completo)
5. ✅ **Métricas** (`.csv` exportado)
6. ✅ **Documentación** (`README.md`)

---

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'tensorflow'"
```bash
# Solución:
pip install tensorflow==2.15.0
```

### Error: "CUDA not found" (GPU)
```bash
# Es normal, el proyecto funciona en CPU
# Para usar GPU:
pip install tensorflow-gpu==2.15.0
# Instalar CUDA Toolkit 11.8 de NVIDIA
```

### Error: "Memory Error" durante entrenamiento
```python
# Solución: Reducir batch size en config.py
BATCH_SIZE = 64  # En lugar de 128
```

### Entrenamiento muy lento
```python
# Solución: Usar main_fast.py o comentar SVM
# SVM es el más lento (5-15 minutos)
```

### Visualizaciones no se generan
```bash
# Verificar que matplotlib funciona:
python -c "import matplotlib.pyplot as plt; plt.plot([1,2,3]); plt.savefig('test.png')"
```

---

## 📚 Referencias y Recursos

### Papers Científicos
- LeCun et al. (1998) - "Gradient-Based Learning Applied to Document Recognition"
- Krizhevsky et al. (2012) - "ImageNet Classification with Deep CNNs"
- Srivastava et al. (2014) - "Dropout: A Simple Way to Prevent Overfitting"

### Documentación Oficial
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Keras API Reference](https://keras.io/api/)

### Datasets
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

---

## 🤝 Contribuciones

Este es un proyecto académico. Para sugerencias o mejoras:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -m 'Agregar mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

---

## 📧 Contacto y Soporte

### Soporte Técnico

**Para problemas comunes:**
1. Revisar logs de consola
2. Verificar instalación de dependencias: `pip list`
3. Comprobar espacio en disco: `df -h` (Linux/Mac) o `dir` (Windows)
4. Consultar sección [Solución de Problemas](#-solución-de-problemas)

**Para errores específicos:**
- Incluir traceback completo del error
- Especificar sistema operativo y versión de Python
- Compartir archivo `pip list > requirements_actual.txt`

---

## 📄 Licencia
```
MIT License

Copyright (c) 2024 TecnoForms - Proyecto Académico

Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia
de este software y archivos de documentación asociados, para usar el Software
sin restricciones, incluyendo sin limitación los derechos de usar, copiar,
modificar, fusionar, publicar, distribuir, sublicenciar y/o vender copias del
Software.

Proyecto desarrollado con fines educativos para el curso de
Machine Learning & Deep Learning.
```

---

## 🎯 Roadmap Futuro

### Versión 2.0 (Planeada)

- [ ] Interfaz gráfica (GUI) con Tkinter/PyQt
- [ ] API REST con FastAPI
- [ ] Deploy en cloud (AWS/Azure/GCP)
- [ ] App móvil con TensorFlow Lite
- [ ] Transfer learning con datos reales de TecnoForms
- [ ] Soporte para múltiples idiomas
- [ ] Dashboard interactivo con Streamlit
- [ ] Detección de confianza en predicciones
- [ ] Sistema de feedback y re-entrenamiento

---

## 🌟 Agradecimientos

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, Christopher Burges
- **TensorFlow Team**: Por framework excepcional
- **Scikit-learn Contributors**: Por herramientas de ML
- **Comunidad de Python**: Por librerías open-source

---

## 📊 Estadísticas del Proyecto
```
📦 Total de líneas de código: ~2,500
🧪 Tests ejecutados: 5 modelos × 10,000 muestras = 50,000 predicciones
⏱️ Horas de desarrollo: ~40-50 horas
📁 Archivos generados: 15+ (modelos, gráficos, reportes)
🎯 Accuracy máxima alcanzada: 99.52% (CNN epoch 18)
```

---

<div align="center">

### 🚀 ¡Listo para Comenzar!
```bash
git clone [tu-repositorio]
cd mnist_digit_classification
pip install -r requirements.txt
python main.py
```

**Desarrollado con ❤️ para TecnoForms**

[⬆ Volver arriba](#-clasificación-de-dígitos-manuscritos---tecnoforms)

**Hecho por el equipo de Hepein**

</div>
