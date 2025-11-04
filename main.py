"""
Script principal para ejecutar el proyecto completo
"""

import warnings
warnings.filterwarnings('ignore')

from preprocessing import DataPreprocessor
from models import ModelBuilder
from train import ModelTrainer
from evaluate import ModelEvaluator
from visualize import ResultVisualizer
from generate_report import ReportGenerator
import config


def main():
    """Función principal"""
    
    print("=" * 80)
    print(" PROYECTO: CLASIFICACIÓN DE DÍGITOS MANUSCRITOS - TECNOFORMS")
    print(" Machine Learning & Deep Learning")
    print("=" * 80)
    print()
    
    # ========================
    # 1. PREPROCESAMIENTO
    # ========================
    print("\n" + "=" * 80)
    print("FASE 1: PREPROCESAMIENTO DE DATOS")
    print("=" * 80)
    
    preprocessor = DataPreprocessor()
    
    # Para modelos de Deep Learning (CNN, ANN)
    X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn = preprocessor.preprocess_pipeline(for_cnn=True)
    
    # Para modelos ML tradicionales (KNN, SVM, RF) Y TAMBIÉN PARA ANN
    X_train_ml, y_train_ml, X_test_ml, y_test_ml = preprocessor.preprocess_pipeline(for_cnn=False)
    
    # ========================
    # 2. CONSTRUCCIÓN DE MODELOS
    # ========================
    print("\n" + "=" * 80)
    print("FASE 2: CONSTRUCCIÓN DE MODELOS")
    print("=" * 80)
    
    builder = ModelBuilder()
    
    # Construir todos los modelos
    cnn_model = builder.build_cnn()
    ann_model = builder.build_ann()
    knn_model = builder.build_knn()
    svm_model = builder.build_svm()
    rf_model = builder.build_random_forest()
    
    # ========================
    # 3. ENTRENAMIENTO
    # ========================
    print("\n" + "=" * 80)
    print("FASE 3: ENTRENAMIENTO DE MODELOS")
    print("=" * 80)
    
    trainer = ModelTrainer()
    
    # Entrenar CNN
    cnn_model, cnn_history = trainer.train_deep_learning_model(
        cnn_model, X_train_cnn, y_train_cnn, 
        model_name='CNN', epochs=config.EPOCHS_CNN
    )
    
    # Entrenar ANN (CORREGIDO - usa datos ML flatten, no CNN)
    # Pero primero necesitamos convertir y_train_ml a categorical
    from tensorflow.keras.utils import to_categorical
    y_train_ml_cat = to_categorical(y_train_ml, 10)
    y_test_ml_cat = to_categorical(y_test_ml, 10)
    
    ann_model, ann_history = trainer.train_deep_learning_model(
        ann_model, X_train_ml, y_train_ml_cat,  # <-- CORREGIDO
        model_name='ANN', epochs=config.EPOCHS_ANN
    )
    
    # Entrenar KNN
    knn_model = trainer.train_ml_model(
        knn_model, X_train_ml, y_train_ml, 
        model_name='KNN'
    )
    
    # Entrenar SVM (puede tomar tiempo)
    print("\n⚠️  ADVERTENCIA: SVM puede tomar varios minutos...")
    svm_model = trainer.train_ml_model(
        svm_model, X_train_ml[:10000], y_train_ml[:10000],  # Muestra reducida para velocidad
        model_name='SVM'
    )
    
    # Entrenar Random Forest
    rf_model = trainer.train_ml_model(
        rf_model, X_train_ml, y_train_ml,
        model_name='RandomForest'
    )
    
    # ========================
    # 4. EVALUACIÓN
    # ========================
    print("\n" + "=" * 80)
    print("FASE 4: EVALUACIÓN DE MODELOS")
    print("=" * 80)
    
    evaluator = ModelEvaluator()
    
    # Evaluar CNN
    cnn_results = evaluator.evaluate_deep_learning_model(
        cnn_model, X_test_cnn, y_test_cnn, model_name='CNN'
    )
    
    # Evaluar ANN (CORREGIDO)
    ann_results = evaluator.evaluate_deep_learning_model(
        ann_model, X_test_ml, y_test_ml_cat, model_name='ANN'  # <-- CORREGIDO
    )
    
    # Evaluar KNN
    knn_results = evaluator.evaluate_ml_model(
        knn_model, X_test_ml, y_test_ml, model_name='KNN'
    )
    
    # Evaluar SVM
    svm_results = evaluator.evaluate_ml_model(
        svm_model, X_test_ml[:2000], y_test_ml[:2000], model_name='SVM'
    )
    
    # Evaluar Random Forest
    rf_results = evaluator.evaluate_ml_model(
        rf_model, X_test_ml, y_test_ml, model_name='RandomForest'
    )
    
    # ========================
    # 5. VISUALIZACIÓN
    # ========================
    print("\n" + "=" * 80)
    print("FASE 5: GENERACIÓN DE VISUALIZACIONES")
    print("=" * 80)
    
    visualizer = ResultVisualizer()
    
    # Historial de entrenamiento
    print("\n📊 Generando gráficos de entrenamiento...")
    visualizer.plot_training_history(cnn_history.history, 'CNN')
    visualizer.plot_training_history(ann_history.history, 'ANN')
    
    # Matrices de confusión
    print("\n📊 Generando matrices de confusión...")
    visualizer.plot_confusion_matrix(cnn_results['confusion_matrix'], 'CNN')
    visualizer.plot_confusion_matrix(ann_results['confusion_matrix'], 'ANN')
    visualizer.plot_confusion_matrix(knn_results['confusion_matrix'], 'KNN')
    
    # Predicciones de muestra
    print("\n📊 Generando visualizaciones de predicciones...")
    visualizer.plot_sample_predictions(
        X_test_cnn, cnn_results['y_true'], cnn_results['y_pred'], 'CNN'
    )
    
    # Comparación de modelos
    print("\n📊 Generando comparación de modelos...")
    all_results = evaluator.get_all_results()
    visualizer.plot_model_comparison(all_results)
    
    # ========================
    # 6. GENERACIÓN DE REPORTES
    # ========================
    print("\n" + "=" * 80)
    print("FASE 6: GENERACIÓN DE REPORTES")
    print("=" * 80)
    
    report_gen = ReportGenerator()
    
    print("\n📝 Generando informes...")
    report_gen.generate_text_report(all_results, trainer.training_history)
    report_gen.generate_csv_report(all_results)
    
    # ========================
    # 7. RESUMEN FINAL
    # ========================
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    
    print("\n🏆 RANKING DE MODELOS:")
    sorted_results = sorted(all_results.items(), 
                            key=lambda x: x[1]['accuracy'], 
                            reverse=True)
    
    for i, (model_name, results) in enumerate(sorted_results, 1):
        print(f"   {i}. {model_name}: {results['accuracy']*100:.2f}% accuracy")
    
    print("\n✅ PROYECTO COMPLETADO EXITOSAMENTE!")
    print(f"\n📁 Resultados guardados en: {config.RESULTS_DIR}")
    print(f"📁 Modelos guardados en: {config.MODELS_DIR}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()