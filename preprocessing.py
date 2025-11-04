"""
Módulo de preprocesamiento de imágenes
"""

import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.datasets import mnist
import config


class DataPreprocessor:
    """Clase para preprocesamiento de datos de dígitos manuscritos"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_mnist_data(self):
        """Carga el dataset MNIST"""
        print("📥 Cargando dataset MNIST...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        print(f"   ✓ Datos de entrenamiento: {X_train.shape}")
        print(f"   ✓ Datos de prueba: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def normalize_images(self, images):
        """Normaliza las imágenes a rango [0, 1]"""
        return images.astype('float32') / 255.0
    
    def reshape_for_cnn(self, images):
        """Reshape para CNN (añade canal)"""
        return images.reshape(images.shape[0], 28, 28, 1)
    
    def reshape_for_ml(self, images):
        """Reshape para modelos ML tradicionales (flatten)"""
        return images.reshape(images.shape[0], -1)
    
    def apply_data_augmentation(self, image):
        """Aplica técnicas de aumento de datos"""
        # Rotación aleatoria
        angle = np.random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((14, 14), angle, 1)
        image = cv2.warpAffine(image, M, (28, 28))
        
        # Traslación aleatoria
        tx = np.random.randint(-2, 3)
        ty = np.random.randint(-2, 3)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (28, 28))
        
        return image
    
    def encode_labels(self, labels, categorical=True):
        """Codifica las etiquetas"""
        if categorical:
            return to_categorical(labels, 10)
        return labels
    
    def preprocess_pipeline(self, for_cnn=True):
        """Pipeline completo de preprocesamiento"""
        print("\n🔄 Iniciando preprocesamiento...")
        
        # Cargar datos
        X_train, y_train, X_test, y_test = self.load_mnist_data()
        
        # Normalizar
        print("   ⚙️ Normalizando imágenes...")
        X_train_norm = self.normalize_images(X_train)
        X_test_norm = self.normalize_images(X_test)
        
        # Reshape según el modelo
        if for_cnn:
            print("   ⚙️ Preparando para CNN...")
            X_train_processed = self.reshape_for_cnn(X_train_norm)
            X_test_processed = self.reshape_for_cnn(X_test_norm)
            y_train_processed = self.encode_labels(y_train, categorical=True)
            y_test_processed = self.encode_labels(y_test, categorical=True)
        else:
            print("   ⚙️ Preparando para modelos ML tradicionales...")
            X_train_processed = self.reshape_for_ml(X_train_norm)
            X_test_processed = self.reshape_for_ml(X_test_norm)
            y_train_processed = y_train
            y_test_processed = y_test
        
        print("   ✅ Preprocesamiento completado!\n")
        
        return X_train_processed, y_train_processed, X_test_processed, y_test_processed