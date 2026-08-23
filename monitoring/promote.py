from mlflow.tracking import MlflowClient

# Connect to your local MLflow tracking server
client = MlflowClient(tracking_uri="http://localhost:5001")

# Force Version 1 of the model into the Production stage
client.transition_model_version_stage(
    name="sentiment-distilbert",
    version=1,
    stage="Production"
)

print("Success! Model is now in Production.")
