# Clasificación de Dígitos Manuscritos - TecnoForms

## 🎯 Descripción del Proyecto

Solución completa de Machine Learning y Deep Learning para la clasificación automática de dígitos manuscritos en formularios. Implementa múltiples algoritmos y genera reportes completos con métricas de rendimiento.

## 🚀 Características

- **5 Modelos Implementados**: CNN, ANN, KNN, SVM, Random Forest
- **Preprocesamiento Completo**: Normalización, segmentación, data augmentation
- **Evaluación Exhaustiva**: Matrices de confusión, métricas detalladas
- **Visualizaciones Profesionales**: Gráficos de alta calidad
- **Reportes Automáticos**: Informes técnicos y CSV con resultados

## 📋 Requisitos

- Python 3.8+
- 4GB RAM mínimo
- 2GB espacio en disco

## 🔧 Instalación

### Paso 1: Clonar o descargar el proyecto
```bash
mkdir mnist_digit_classification
cd mnist_digit_classification
```

### Paso 2: Crear entorno virtual (recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar dependencias
```bash
pip install -r requirements.txt
```
o este con ultima version
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python Pillow joblib tf-keras
```
## ▶️ Ejecución

### Ejecutar proyecto completo
```bash
python main.py
```

El proyecto ejecutará automáticamente:
1. Carga y preprocesamiento de datos
2. Construcción de 5 modelos diferentes
3. Entrenamiento de todos los modelos
4. Evaluación con métricas completas
5. Generación de visualizaciones
6. Creación de reportes

### Tiempo estimado de ejecución
- Total: ~15-25 minutos
- CNN: ~5-8 minutos
- ANN: ~3-5 minutos
- KNN: ~1-2 minutos
- SVM: ~5-10 minutos
- Random Forest: ~1-2 minutos

## 📂 Estructura de Salidas
```
mnist_digit_classification/
├── models/              # Modelos entrenados (.h5, .pkl)
├── results/
│   ├── figures/        # Gráficos y visualizaciones
│   └── reports/        # Informes técnicos y CSV
```

## 📊 Resultados Esperados

### Precisión Aproximada por Modelo:
- **CNN**: ~99.0-99.5%
- **ANN**: ~97.5-98.5%
- **Random Forest**: ~96.5-97.5%
- **KNN**: ~96.0-97.0%
- **SVM**: ~94.0-95.0%

## 📈 Métricas Generadas

- Accuracy, Precision, Recall, F1-Score
- Matrices de confusión
- Curvas de aprendizaje
- Comparación entre modelos
- Reporte de clasificación por dígito

## 🛠️ Personalización

### Modificar hiperparámetros
Editar `config.py`:
```python
EPOCHS_CNN = 20        # Épocas para CNN
BATCH_SIZE = 128       # Tamaño de batch
KNN_NEIGHBORS = 5      # Vecinos para KNN
```

### Entrenar un solo modelo
Modificar `main.py` comentando modelos no deseados.

## 📝 Arquitectura de la CNN
```
Input (28x28x1)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → Dropout(0.3)
    ↓
Output Dense(10) + Softmax
```

## 🎓 Uso Académico

Este proyecto cumple con los requisitos de:
- Selección y comparación de algoritmos
- Preprocesamiento de imágenes
- Diseño de redes neuronales
- Validación y ajuste de hiperparámetros
- Evaluación con métricas estándar
- Generación de reportes técnicos

## 📧 Soporte

Para dudas o problemas:
- Revisar logs de consola
- Verificar instalación de dependencias
- Comprobar espacio en disco

## 📄 Licencia

Proyecto académico - Uso educativo

---
**TecnoForms** - Machine Learning & Deep Learning Course
