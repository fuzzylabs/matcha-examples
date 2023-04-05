#!/bin/bash
echo "Installing example requirements (see requirements.txt)..."
{
    pip install -r requirements.txt
    
    zenml integration install mlflow azure kubernetes -y
} >> setup_out.log

if [[ ! -f .matcha/infrastructure/matcha.state ]]
then
    echo "Error: The file .matcha/infrastructure/matcha.state does not exist!"
    echo "Ensure that you have run 'matcha provision' in this directory and all cloud resources have been provisioned."
    exit 1
fi

mlflow_tracking_url=$(sed -n 's/.*"mlflow-tracking-url": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)
zenml_storage_path=$(sed -n 's/.*"zenml-storage-path": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)
zenml_connection_string=$(sed -n 's/.*"zenml-connection-string": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)
k8s_context=$(sed -n 's/.*"k8s-context": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)
acr_registry_uri=$(sed -n 's/.*"azure-container-registry": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)
acr_registry_name=$(sed -n 's/.*"azure-container-name": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)
zenserver_url=$(sed -n 's/.*"zen-server-url": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)
zenserver_username=$(sed -n 's/.*"zen-server-username": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)
zenserver_password=$(sed -n 's/.*"zen-server-password": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state) 

echo "Setting up ZenML..."
{
    export AUTO_OPEN_DASHBOARD=false
    export AZURE_STORAGE_CONNECTION_STRING="$zenml_connection_string"

    az acr login --name="$acr_registry_name"

    zenml init 
    zenml connect --url="$zenserver_url" --username="$zenserver_username" --password="$zenserver_password" --no-verify-ssl
    zenml container-registry register acr_registry -f azure --uri="$acr_registry_uri"
    zenml experiment-tracker register mlflow_experiment_tracker -f mlflow --tracking_uri="$mlflow_tracking_url" --tracking_username=username --tracking_password=password
    zenml artifact-store register az_store -f azure --path="$zenml_storage_path"
    zenml orchestrator register k8s_orchestrator -f kubernetes --kubernetes_context="$k8s_context" --kubernetes_namespace=zenml --synchronous=True
    zenml stack register recommendation_example_cloud_stack -c acr_registry -e mlflow_experiment_tracker -a az_store -o k8s_orchestrator --set
} >> setup_out.log
