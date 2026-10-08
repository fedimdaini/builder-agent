import os
import mlflow
import mlflow.xgboost
from src.models.train_model import train_xgboost
import pandas as pd

def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    
    # Set experiment
    experiment_name = "taxi-duration"
    mlflow.set_experiment(experiment_name)
    
    # Load training data
    train_path = "data/processed/train.csv"
    target_col = "duration"
    
    # Sample mode
    sample_mode = os.getenv("SAMPLE", "0") == "1"
    if sample_mode:
        df_train = pd.read_csv(train_path).sample(frac=0.01, random_state=42)
    else:
        df_train = pd.read_csv(train_path)
    
    # Train the model
    model = train_xgboost(train_path, target_col)
    
    # Log model and metrics
    with mlflow.start_run():
        mlflow.xgboost.log_model(model, "model")
        Y_train = np.log1p(df_train[target_col].values)
        X_train = df_train.drop(target_col, axis=1).values
        dmatrix_train = xgb.DMatrix(X_train, label=Y_train)
        predictions = model.predict(dmatrix_train)
        predictions = np.expm1(predictions)
        Y_train = np.expm1(Y_train)
        rmse = np.sqrt(np.mean((predictions - Y_train) ** 2))
        mae = np.mean(np.abs(predictions - Y_train))
        r2 = 1 - np.sum((predictions - Y_train) ** 2) / np.sum((Y_train - np.mean(Y_train)) ** 2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

if __name__ == "__main__":
    main()
