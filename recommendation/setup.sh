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

echo "Setting up ZenML..."
{
    export AUTO_OPEN_DASHBOARD=false
    export AZURE_STORAGE_CONNECTION_STRING="$zenml_connection_string"

    zenml init 
    # zenml connect --url=https://35.246.120.161.nip.io/ --username=default --password=wKAUwApldakpiF2G --no-verify-ssl
    zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow --tracking_uri="$mlflow_tracking_url" --tracking_username=username --tracking_password=password
    zenml artifact-store register az_store -f azure --path="$zenml_storage_path"
    zenml orchestrator register k8s_orchestrator --flavor=kubernetes --kubernetes_context="$k8s_context" --kubernetes_namespace=zenml --synchronous=True
    zenml stack register recommendation_example_stack -e mlflow_experiment_tracker -a az_store -o k8s_orchestrator --set
} >> setup_out.log
