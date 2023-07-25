#!/bin/bash
echo "Installing example requirements (see requirements.txt)..."
{
    pip install -r requirements.txt
    pip install "cython<3" wheel setuptools && pip install --no-build-isolation "pyyaml==5.4.1"
    pip install "zenml[server]==0.36.1"
    pip install --pre --extra-index https://pypi.anaconda.org/scipy-wheels-nightly/simple "scikit-surprise==1.1.3"
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
    json_string=$(matcha get "$resource_name" "$property" --output json --show-sensitive)
    value=$(echo "$json_string" | sed -n 's/.*"'$property'": "\(.*\)".*/\1/p')
    echo "$value"
}

mlflow_tracking_url=$(get_state_value experiment-tracker tracking-url)
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
    
    # Disconnect from previous server if exists to prevent errors when a new Zen server is being used
    zenml disconnect

    zenml connect --url="$zenserver_url" --username="$zenserver_username" --password="$zenserver_password" --no-verify-ssl

    zenml secret create az_secret --connection_string="$zenml_connection_string"
    zenml container-registry register acr_registry -f azure --uri="$acr_registry_uri"
    zenml experiment-tracker register mlflow_experiment_tracker -f mlflow --tracking_uri="$mlflow_tracking_url" --tracking_username=username --tracking_password=password
    zenml artifact-store register az_store -f azure --path="$zenml_storage_path" --authentication_secret=az_secret
    zenml orchestrator register k8s_orchestrator -f kubernetes --kubernetes_context="$k8s_context" --kubernetes_namespace=zenml --synchronous=True
    zenml image-builder register docker_builder --flavor=local

    # Register the Seldon Core Model Deployer
    zenml model-deployer register seldon_deployer --flavor=seldon \
        --kubernetes_context=$k8s_context \
        --kubernetes_namespace=$seldon_workload_namespace \
        --base_url=http://$seldon_ingress_host \

    zenml stack register recommendation_example_cloud_stack -i docker_builder -c acr_registry -e mlflow_experiment_tracker -a az_store -o k8s_orchestrator --model_deployer=seldon_deployer --set
} >> setup_out.log

echo "ZenML set-up complete."
