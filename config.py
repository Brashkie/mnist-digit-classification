"""
Configuración global del proyecto
"""

import os

# Directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')

# Crear directorios si no existen
for directory in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Parámetros de preprocesamiento
IMAGE_SIZE = (28, 28)
GRAYSCALE = True
NORMALIZE = True

# Parámetros de entrenamiento
BATCH_SIZE = 128
EPOCHS_CNN = 20
EPOCHS_ANN = 15
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# Parámetros de modelos
# CNN
CNN_FILTERS = [32, 64, 128]
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)
CNN_DROPOUT = 0.5

# ANN (Red Neuronal Densa)
ANN_LAYERS = [512, 256, 128]
ANN_DROPOUT = 0.3

# Otros modelos ML
KNN_NEIGHBORS = 5
SVM_C = 10
SVM_KERNEL = 'rbf'

# Métricas
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']