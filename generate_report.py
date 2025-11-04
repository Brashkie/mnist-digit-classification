"""
Módulo de generación de reportes
"""

import pandas as pd
from datetime import datetime
import config


class ReportGenerator:
    """Clase para generación de reportes"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_text_report(self, results_dict, training_history):
        """Genera reporte en formato texto"""
        report_path = f"{config.REPORTS_DIR}/informe_completo_{self.timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("INFORME DE CLASIFICACIÓN DE DÍGITOS MANUSCRITOS\n")
            f.write("TecnoForms - Machine Learning & Deep Learning\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Introducción
            f.write("1. INTRODUCCIÓN\n")
            f.write("-" * 80 + "\n")
            f.write("Este proyecto implementa una solución de Machine Learning para el\n")
            f.write("reconocimiento automático de dígitos manuscritos en formularios.\n")
            f.write("Se utilizó el dataset MNIST y se compararon múltiples algoritmos.\n\n")
            
            # Metodología
            f.write("2. METODOLOGÍA\n")
            f.write("-" * 80 + "\n")
            f.write("Dataset: MNIST (70,000 imágenes de dígitos manuscritos)\n")
            f.write("- Entrenamiento: 60,000 imágenes\n")
            f.write("- Prueba: 10,000 imágenes\n")
            f.write("- Resolución: 28x28 píxeles en escala de grises\n\n")
            
            f.write("Preprocesamiento:\n")
            f.write("- Normalización de píxeles [0, 1]\n")
            f.write("- Conversión a escala de grises\n")
            f.write("- Segmentación automática\n\n")
            
            # Modelos implementados
            f.write("3. MODELOS IMPLEMENTADOS\n")
            f.write("-" * 80 + "\n")
            for model_name in results_dict.keys():
                f.write(f"✓ {model_name}\n")
            f.write("\n")
            
            # Resultados por modelo
            f.write("4. RESULTADOS DETALLADOS\n")
            f.write("-" * 80 + "\n\n")
            
            for model_name, results in results_dict.items():
                f.write(f"4.{list(results_dict.keys()).index(model_name) + 1} {model_name.upper()}\n")
                f.write("-" * 40 + "\n")
                
                f.write(f"Accuracy:  {results['accuracy']*100:.2f}%\n")
                f.write(f"Precision: {results['precision']*100:.2f}%\n")
                f.write(f"Recall:    {results['recall']*100:.2f}%\n")
                f.write(f"F1-Score:  {results['f1_score']*100:.2f}%\n\n")
                
                if model_name in training_history:
                    time = training_history[model_name]['training_time']
                    f.write(f"Tiempo de entrenamiento: {time:.2f} segundos\n\n")
            
            # Comparación
            f.write("5. COMPARACIÓN DE MODELOS\n")
            f.write("-" * 80 + "\n")
            
            # Encontrar mejor modelo
            best_model = max(results_dict.items(), 
                           key=lambda x: x[1]['accuracy'])
            
            f.write(f"Mejor modelo: {best_model[0]}\n")
            f.write(f"Accuracy: {best_model[1]['accuracy']*100:.2f}%\n\n")
            
            # Tabla comparativa
            f.write("Tabla Comparativa:\n")
            f.write(f"{'Modelo':<20} {'Accuracy':<12} {'F1-Score':<12}\n")
            f.write("-" * 44 + "\n")
            
            for model_name, results in sorted(results_dict.items(), 
                                            key=lambda x: x[1]['accuracy'], 
                                            reverse=True):
                f.write(f"{model_name:<20} {results['accuracy']*100:>10.2f}%  "
                       f"{results['f1_score']*100:>10.2f}%\n")
            
            f.write("\n")
            
            # Conclusiones
            f.write("6. CONCLUSIONES\n")
            f.write("-" * 80 + "\n")
            f.write(f"• El modelo {best_model[0]} obtuvo el mejor rendimiento\n")
            f.write(f"• Se logró una precisión del {best_model[1]['accuracy']*100:.2f}%\n")
            f.write("• La solución es viable para implementación en producción\n")
            f.write("• Se recomienda usar CNN para máxima precisión\n\n")
            
            # Recomendaciones
            f.write("7. RECOMENDACIONES\n")
            f.write("-" * 80 + "\n")
            f.write("• Implementar el modelo CNN en producción\n")
            f.write("• Realizar validación con formularios reales\n")
            f.write("• Implementar sistema de retroalimentación\n")
            f.write("• Monitorear rendimiento continuo\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("FIN DEL INFORME\n")
            f.write("=" * 80 + "\n")
        
        print(f"   📄 Informe generado: {report_path}")
        
        return report_path
    
    def generate_csv_report(self, results_dict):
        """Genera reporte en formato CSV"""
        data = []
        
        for model_name, results in results_dict.items():
            data.append({
                'Modelo': model_name,
                'Accuracy (%)': f"{results['accuracy']*100:.2f}",
                'Precision (%)': f"{results['precision']*100:.2f}",
                'Recall (%)': f"{results['recall']*100:.2f}",
                'F1-Score (%)': f"{results['f1_score']*100:.2f}"
            })
        
        df = pd.DataFrame(data)
        csv_path = f"{config.REPORTS_DIR}/metricas_{self.timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"   📊 Métricas guardadas: {csv_path}")
        
        return csv_path