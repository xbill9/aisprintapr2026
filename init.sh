#!/bin/bash

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "Error: No active gcloud account found."
    echo "Please run 'gcloud auth login' and try again."
    exit 1
fi

if [ -f "$HOME/project_id.txt" ]; then
    PROJECT_ID=$(cat "$HOME/project_id.txt")
else
    read -p "Enter Project ID: " PROJECT_ID
    echo "$PROJECT_ID" > "$HOME/project_id.txt"
fi

gcloud config set project "$PROJECT_ID"

gcloud services enable aiplatform.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudaicompanion.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable storage.googleapis.com

# Grant IAM roles to the current user
ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="user:$ACTIVE_ACCOUNT" --role="roles/logging.admin" --quiet
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="user:$ACTIVE_ACCOUNT" --role="roles/storage.admin" --quiet

# Grant Logging Viewer and View Accessor roles to the Default Compute Service Account
# This allows the Cloud Run service to fetch and analyze logs.
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/logging.viewer" --quiet
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/logging.viewAccessor" --quiet

# Create Cloud Storage bucket
gcloud storage buckets create gs://"$PROJECT_ID"-bucket --location=us-east4 || true


#curl -s https://raw.githubusercontent.com/haren-bh/gcpbillingactivate/main/activate.py | python3

cat <<EOF > .env
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_CLOUD_LOCATION=us-east4
IMAGEN_MODEL="imagen-3.0-fast-generate-001"
GENAI_MODEL="gemini-2.5-flash"
BUCKET_NAME="$PROJECT_ID-bucket"
EOF

source .env

if [ -z "$CLOUD_SHELL" ]; then
    if ! gcloud auth application-default print-access-token > /dev/null 2>&1; then
        echo "ADC expired or not found. Initializing login..."
        gcloud auth application-default login
    else
        echo "ADC is valid."
    fi
fi

if [ ! -f ".requirements_installed" ]; then
    pip install -r requirements.txt
    touch .requirements_installed
fi

echo "Environment setup"
cat .env

echo "Cloud Login"
gcloud auth list

echo "ADK update"
pip install google-adk --upgrade
adk --version
