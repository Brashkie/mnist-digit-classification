"""
Módulo de visualización de resultados
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config


class ResultVisualizer:
    """Clase para visualización de resultados"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(self, history, model_name='model'):
        """Grafica el historial de entrenamiento"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/{model_name}_training_history.png', 
                    dpi=300, bbox_inches='tight')
        print(f"   💾 Gráfico guardado: {model_name}_training_history.png")
        plt.close()
    
    def plot_confusion_matrix(self, cm, model_name='model'):
        """Grafica la matriz de confusión"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=config.CLASS_NAMES,
                    yticklabels=config.CLASS_NAMES,
                    cbar_kws={'label': 'Cantidad'})
        plt.title(f'Matriz de Confusión - {model_name}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Etiqueta Real', fontsize=12)
        plt.xlabel('Etiqueta Predicha', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/{model_name}_confusion_matrix.png', 
                    dpi=300, bbox_inches='tight')
        print(f"   💾 Gráfico guardado: {model_name}_confusion_matrix.png")
        plt.close()
    
    def plot_sample_predictions(self, X_test, y_true, y_pred, model_name='model', n_samples=20):
        """Visualiza predicciones de muestra"""
        # Seleccionar muestras aleatorias
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Obtener imagen
            if len(X_test.shape) == 4:  # CNN format
                img = X_test[idx].reshape(28, 28)
            else:  # Flattened format
                img = X_test[idx].reshape(28, 28)
            
            # Graficar
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            
            # Color según si es correcto o no
            color = 'green' if y_true[idx] == y_pred[idx] else 'red'
            axes[i].set_title(f'Real: {y_true[idx]}\nPred: {y_pred[idx]}', 
                             color=color, fontweight='bold')
        
        plt.suptitle(f'Predicciones de Muestra - {model_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/{model_name}_sample_predictions.png', 
                    dpi=300, bbox_inches='tight')
        print(f"   💾 Gráfico guardado: {model_name}_sample_predictions.png")
        plt.close()
    
    def plot_model_comparison(self, results_dict):
        """Compara métricas de diferentes modelos"""
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results_dict[model][metric] * 100 for model in models]
            ax.bar(x + i * width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
        ax.set_ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
        ax.set_title('Comparación de Modelos', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/model_comparison.png', 
                    dpi=300, bbox_inches='tight')
        print(f"   💾 Gráfico guardado: model_comparison.png")
        plt.close()