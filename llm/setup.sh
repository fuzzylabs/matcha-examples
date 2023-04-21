#!/bin/bash
echo "Installing example requirements (see requirements.txt)..."
{
    pip install -r requirements.txt
    
    zenml integration install azure huggingface pytorch seldon kubernetes -y
} >> setup_out.log

if [[ ! -f .matcha/infrastructure/matcha.state ]]
then
    echo "Error: The file .matcha/infrastructure/matcha.state does not exist!"
    echo "Ensure that you have run 'matcha provision' in this directory and all cloud resources have been provisioned."
    exit 1
fi

get_state_value() {
    key=$1
    value=$(sed -n 's/.*"'$key'": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)
    if [[ -z $value ]]; then
        echo "Error: The value for '$key' is not found in .matcha/infrastructure/matcha.state!"
        exit 1
    fi
    echo $value
}

zenml_storage_path=$(get_state_value zenml_storage_path)
zenml_connection_string=$(get_state_value zenml_connection_string)
k8s_context=$(get_state_value k8s_context)
acr_registry_uri=$(get_state_value azure_container_registry)
acr_registry_name=$(get_state_value azure_registry_name)
zenserver_url=$(get_state_value zen_server_url)
zenserver_username=$(get_state_value zen_server_username)
zenserver_password=$(get_state_value zen_server_password)
seldon_workload_namespace=$(get_state_value seldon_workloads_namespace)
seldon_ingress_host=$(get_state_value seldon_base_url)

zenml_storage_container="${zenml_storage_path##*/}"
echo "AZURE_STORAGE_CONNECTION_STRING=\"$zenml_connection_string\"" > "./server/.env"
echo "AZURE_STORAGE_CONTAINER_NAME=\"$zenml_storage_container\"" >> "./server/.env"


echo "Setting up ZenML..."
{
    export AUTO_OPEN_DASHBOARD=false
    az acr login --name="$acr_registry_name"

    zenml init
    zenml connect --url="$zenserver_url" --username="$zenserver_username" --password="$zenserver_password" --no-verify-ssl
    zenml secret create az_secret --connection_string="$zenml_connection_string"
    zenml image-builder register docker_builder --flavor=local
    zenml container-registry register acr_registry -f azure --uri="$acr_registry_uri"
    zenml artifact-store register az_store -f azure --path="$zenml_storage_path" --authentication_secret=az_secret
    zenml model-deployer register seldon_deployer --flavor=seldon \
        --kubernetes_context=$k8s_context \
        --kubernetes_namespace=$seldon_workload_namespace \
        --base_url=http://$seldon_ingress_host \

    zenml stack register llm_example_cloud_stack -i docker_builder -c acr_registry -a az_store -o default --model_deployer=seldon_deployer --set
} >> setup_out.log