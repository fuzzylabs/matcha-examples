#!/bin/bash
echo "Installing example requirements (see requirements.txt)..."
{
    # Install jq on macOS using Homebrew
    if [[ "$(uname -s)" == "Darwin" ]]; then
        if ! command -v brew &> /dev/null; then
            echo "Error: Homebrew is not installed."
            exit 1
        fi

        if ! command -v jq &> /dev/null; then
            echo "Installing jq using Homebrew..."
            brew install jq
        fi
    fi

    # Install jq on Linux using APT
    if [[ "$(uname -s)" == "Linux" ]]; then
        if ! command -v apt-get &> /dev/null; then
            echo "Error: APT is not available."
            exit 1
        fi

        if ! command -v jq &> /dev/null; then
            echo "Installing jq using APT..."
            sudo apt-get update
            sudo apt-get install -y jq
        fi
    fi

    pip install -r requirements.txt
    zenml integration install mlflow azure kubernetes seldon -y
} >> setup_out.log

if [[ ! -f .matcha/infrastructure/matcha.state ]]
then
    echo "Error: The file .matcha/infrastructure/matcha.state does not exist!"
    echo "Ensure that you have run 'matcha provision' in this directory and all cloud resources have been provisioned."
    exit 1
fi


function get_state_value() {
    resource_name=$1
    property=$2
    json_string=$(matcha get $resource_name $property --output json)
    value=$(echo $json_string | jq -r '."'$resource_name'"."'$property'"')
    echo $value
}

zenml_storage_path=$(get_state_value pipeline storage-path)
zenml_connection_string=$(get_state_value pipeline connection-string)
k8s_context=$(get_state_value orchestrator k8s-context)
acr_registry_uri=$(get_state_value container-registry registry-url)
acr_registry_name=$(get_state_value container-registry registry-name)
zenserver_url=$(get_state_value pipeline server-url)
zenserver_username=$(get_state_value pipeline server-username)
zenserver_password=$(get_state_value pipeline server-password)
seldon_workload_namespace=$(get_state_value model-deployer workloads-namespace)
seldon_ingress_host=$(get_state_value model-deployer base-url)


echo "Setting up ZenML..."
{
    export AUTO_OPEN_DASHBOARD=false
    az acr login --name="$acr_registry_name"

    zenml init

    zenml connect --url="$zenserver_url" --username="$zenserver_username" --password="$zenserver_password" --no-verify-ssl
    zenml secret create az_secret --connection_string="$zenml_connection_string"
    zenml container-registry register acr_registry -f azure --uri="$acr_registry_uri"
    zenml artifact-store register az_store -f azure --path="$zenml_storage_path" --authentication_secret=az_secret
    zenml orchestrator register k8s_orchestrator -f kubernetes --kubernetes_context="$k8s_context" --kubernetes_namespace=zenml --synchronous=True
    zenml image-builder register docker_builder --flavor=local

    # Register the Seldon Core Model Deployer
    zenml model-deployer register seldon_deployer --flavor=seldon \
        --kubernetes_context=$k8s_context \
        --kubernetes_namespace=$seldon_workload_namespace \
        --base_url=http://$seldon_ingress_host \

    zenml stack register llm_example_cloud_stack -i docker_builder -c acr_registry -a az_store -o k8s_orchestrator --model_deployer=seldon_deployer --set
} >> setup_out.log

echo "ZenML set-up complete."
