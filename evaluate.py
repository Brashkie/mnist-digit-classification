"""
Módulo de evaluación de modelos
"""

import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score)
import config


class ModelEvaluator:
    """Clase para evaluación de modelos"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_deep_learning_model(self, model, X_test, y_test, model_name='model'):
        """Evalúa modelos de Deep Learning"""
        print(f"\n📊 Evaluando {model_name}...")
        
        # Predicciones
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Evaluar
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Métricas adicionales
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        # Reporte de clasificación
        report = classification_report(y_true, y_pred, 
                                       target_names=config.CLASS_NAMES,
                                       output_dict=True)
        
        results = {
            'model_name': model_name,
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        self.results[model_name] = results
        
        print(f"   📈 Accuracy: {accuracy*100:.2f}%")
        print(f"   📈 Precision: {precision*100:.2f}%")
        print(f"   📈 Recall: {recall*100:.2f}%")
        print(f"   📈 F1-Score: {f1*100:.2f}%")
        
        return results
    
    def evaluate_ml_model(self, model, X_test, y_test, model_name='model'):
        """Evalúa modelos de Machine Learning tradicionales"""
        print(f"\n📊 Evaluando {model_name}...")
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_true = y_test
        
        # Métricas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        # Reporte de clasificación
        report = classification_report(y_true, y_pred, 
                                       target_names=config.CLASS_NAMES,
                                       output_dict=True)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        self.results[model_name] = results
        
        print(f"   📈 Accuracy: {accuracy*100:.2f}%")
        print(f"   📈 Precision: {precision*100:.2f}%")
        print(f"   📈 Recall: {recall*100:.2f}%")
        print(f"   📈 F1-Score: {f1*100:.2f}%")
        
        return results
    
    def get_all_results(self):
        """Obtiene todos los resultados de evaluación"""
        return self.results