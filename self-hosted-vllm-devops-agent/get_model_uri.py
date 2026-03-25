from google.cloud import aiplatform_v1

def get_model_artifact_uri(project_id, publisher="google", model="gemma-2b-it"):
    from google.cloud.aiplatform_v1.services.model_garden_service import ModelGardenServiceClient
    
    client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    client = ModelGardenServiceClient(client_options=client_options)
    
    # Format the publisher model name
    name = f"publishers/{publisher}/models/{model}"
    
    try:
        response = client.get_publisher_model(name=name)
        print(f"Model: {response.display_name}")
        # Print all available information to find the artifact URI
        # Predict schemata often contains artifact info
        print(f"Predict Schemata: {response.predict_schemata}")
        # Also check open_source_category
        print(f"Open Source Category: {response.open_source_category}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_model_artifact_uri("aisprint-491218")
