import mlflow
import mlflow.sklearn
import os

# import our external modules
from evaluation import model_evaluate
from model_training import model_training
from prepocessing import load_data_frame, preprocessing_data

# Main function
def main():
    # Defines the directory of our file
    df_path = "C:/Users/palom/OneDrive/Dokumen/CUCEA/2DO_SEM/Programacion-2/Challenge_1/data/breast-cancer-wisconsin.data.csv"
    # Load the data
    df = load_data_frame(path=df_path)
    # Preprocessing part
    X, y = preprocessing_data(df=df, scaling=True)

    # Model training
    X_train, X_test, y_train, y_test, model, y_pred = model_training(X=X, y=y)
    
    # Model evaluation
    accuracy, report = model_evaluate(model, X_train, y_train, X_test, y_test, y_pred)

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

# Main Execution Block: Code that runs when the script is executed directly
if __name__ == '__main__':
    main()