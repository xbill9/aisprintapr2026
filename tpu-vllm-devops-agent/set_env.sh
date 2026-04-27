#!/bin/bash

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "Error: No active gcloud account found."
    echo "Please run 'gcloud auth login' and try again."
fi

# Get current project
if [ -f "$HOME/project_id.txt" ]; then
    PROJECT_ID=$(cat "$HOME/project_id.txt")
fi

if [ "$PROJECT_ID" == "(unset)" ] || [ -z "$PROJECT_ID" ]; then
    echo "Warning: No gcloud project is currently set."
    echo "Run 'gcloud config set project [PROJECT_ID]' to configure it."
fi

gcloud config set project $PROJECT_ID


if [ -f "$HOME/gemini.key" ]; then
    GOOGLE_API_KEY=$(cat "$HOME/gemini.key")
else
    read -p "Enter Gemini KEY: " GOOGLE_API_KEY
    echo "$GOOGLE_API_KEY" > "$HOME/gemini.key"
fi

cat <<EOF > .env
GOOGLE_GENAI_USE_VERTEXAI=false
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_CLOUD_LOCATION=southamerica-east1
GOOGLE_CLOUD_REGION=southamerica-east1
GOOGLE_CLOUD_ZONE=southamerica-east1-c
MODEL=google/gemma-4-31B-it
GENAI_MODEL="gemini-2.5-flash"
GOOGLE_API_KEY=$GOOGLE_API_KEY
GEMINI_API_KEY=$GOOGLE_API_KEY
EOF

source .env

echo "Current Environment"
cat .env

echo "Cloud Login"
gcloud auth list

gcloud config list

echo "ADK Version"
adk --version
