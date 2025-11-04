"""
Definición de modelos de Machine Learning y Deep Learning
"""

import numpy as np
from keras.models import Sequential
from keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten, 
                            Dropout, BatchNormalization)
from keras.optimizers import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import config


class ModelBuilder:
    """Clase para construcción de diferentes modelos"""
    
    @staticmethod
    def build_cnn():
        """Construye una Red Neuronal Convolucional (CNN)"""
        print("🏗️ Construyendo CNN...")
        
        model = Sequential([
            # Primera capa convolucional
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), 
                   padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Segunda capa convolucional
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Tercera capa convolucional
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Capas densas
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("   ✅ CNN construida exitosamente!")
        print(f"   📊 Parámetros totales: {model.count_params():,}")
        
        return model
    
    @staticmethod
    def build_ann():
        """Construye una Red Neuronal Artificial (ANN) densa"""
        print("🏗️ Construyendo ANN...")
        
        model = Sequential([
            Dense(512, activation='relu', input_shape=(784,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("   ✅ ANN construida exitosamente!")
        print(f"   📊 Parámetros totales: {model.count_params():,}")
        
        return model
    
    @staticmethod
    def build_knn():
        """Construye modelo K-Nearest Neighbors"""
        print("🏗️ Construyendo KNN...")
        model = KNeighborsClassifier(
            n_neighbors=config.KNN_NEIGHBORS,
            n_jobs=-1
        )
        print("   ✅ KNN construido exitosamente!")
        return model
    
    @staticmethod
    def build_svm():
        """Construye modelo Support Vector Machine"""
        print("🏗️ Construyendo SVM...")
        model = SVC(
            C=config.SVM_C,
            kernel=config.SVM_KERNEL,
            gamma='scale',
            random_state=config.RANDOM_STATE
        )
        print("   ✅ SVM construido exitosamente!")
        return model
    
    @staticmethod
    def build_random_forest():
        """Construye modelo Random Forest"""
        print("🏗️ Construyendo Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        print("   ✅ Random Forest construido exitosamente!")
        return model