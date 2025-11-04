"""
Módulo de entrenamiento de modelos
"""

import time
import joblib
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import config


class ModelTrainer:
    """Clase para entrenamiento de modelos"""
    
    def __init__(self):
        self.training_history = {}
    
    def train_deep_learning_model(self, model, X_train, y_train, 
                                    model_name='model', epochs=20):
        """Entrena modelos de Deep Learning (CNN, ANN)"""
        print(f"\n🚀 Entrenando {model_name}...")
        print(f"   📊 Épocas: {epochs}")
        print(f"   📊 Batch size: {config.BATCH_SIZE}")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        # Entrenar
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=epochs,
            validation_split=config.VALIDATION_SPLIT,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"   ✅ Entrenamiento completado en {training_time:.2f} segundos")
        
        # Guardar modelo
        model_path = f"{config.MODELS_DIR}/{model_name}.keras"
        model.save(model_path)
        print(f"   💾 Modelo guardado en: {model_path}")
        
        self.training_history[model_name] = {
            'history': history.history,
            'training_time': training_time
        }
        
        return model, history
    
    def train_ml_model(self, model, X_train, y_train, model_name='model'):
        """Entrena modelos de Machine Learning tradicionales"""
        print(f"\n🚀 Entrenando {model_name}...")
        
        start_time = time.time()
        
        # Entrenar
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        print(f"   ✅ Entrenamiento completado en {training_time:.2f} segundos")
        
        # Guardar modelo
        model_path = f"{config.MODELS_DIR}/{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"   💾 Modelo guardado en: {model_path}")
        
        self.training_history[model_name] = {
            'training_time': training_time
        }
        
        return model