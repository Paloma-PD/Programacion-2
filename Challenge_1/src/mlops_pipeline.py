import mlflow
import mlflow.sklearn

def model_mlflow(model, X_train, y_train, X_test, y_test):
    # Starting an experiment in MLflow
    mlflow.set_experiment("Breast Cancer Wisconsin")
    with mlflow.start_run():
        # Record metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")
        
        # Record the ROC curve as an image
        mlflow.log_artifact("roc_curve.png")

        # # Record the confusion matrix as an image
        mlflow.log_artifact("confusion_matrix.png")
        
        # Record the model
        mlflow.sklearn.log_model(model, "random_forest_model")