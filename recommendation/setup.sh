#!/bin/bash
echo "Installing example requirements (see requirements.txt)..."
{
    pip install -r requirements.txt
    
    zenml integration install mlflow azure kubernetes seldon -y
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

mlflow_tracking_url=$(get_state_value mlflow-tracking-url)
zenml_storage_path=$(get_state_value zenml-storage-path)
zenml_connection_string=$(get_state_value zenml-connection-string)
k8s_context=$(get_state_value k8s-context)
acr_registry_uri=$(get_state_value azure-container-registry)
acr_registry_name=$(get_state_value azure-registry-name)
zenserver_url=$(get_state_value zen-server-url)
zenserver_username=$(get_state_value zen-server-username)
zenserver_password=$(get_state_value zen-server-password)
seldon_workload_namespace=$(get_state_value seldon-workloads-namespace)
seldon_ingress_host=$(get_state_value seldon-base-url)

# A list of env to check
variables=(
  mlflow_tracking_url
  zenml_storage_path
  zenml_connection_string
  k8s_context
  acr_registry_uri
  acr_registry_name
  zenserver_url
  zenserver_username
  zenserver_password
  seldon_workload_namespace
  seldon_ingress_host
)

# Loop through the array and check whether a variable is empty
for variable in "${variables[@]}"; do
  value=${!variable}
  if [ -z "$value" ]; then
    echo "One or more required variable is empty, please try running matcha provision again."
    break
  fi
done


echo "Setting up ZenML..."
{
    export AUTO_OPEN_DASHBOARD=false
    az acr login --name="$acr_registry_name"

    zenml init
    
    zenml connect --url="$zenserver_url" --username="$zenserver_username" --password="$zenserver_password" --no-verify-ssl
    zenml secret create az_secret --connection_string="$zenml_connection_string"
    zenml container-registry register acr_registry -f azure --uri="$acr_registry_uri"
    zenml experiment-tracker register mlflow_experiment_tracker -f mlflow --tracking_uri="$mlflow_tracking_url" --tracking_username=username --tracking_password=password
    zenml artifact-store register az_store -f azure --path="$zenml_storage_path" --authentication_secret=az_secret
    zenml orchestrator register k8s_orchestrator -f kubernetes --kubernetes_context="$k8s_context" --kubernetes_namespace=zenml --synchronous=True

    # Register the Seldon Core Model Deployer
    zenml model-deployer register seldon_deployer --flavor=seldon \
        --kubernetes_context=$k8s_context \
        --kubernetes_namespace=$seldon_workload_namespace \
        --base_url=http://$seldon_ingress_host \

    zenml stack register recommendation_example_cloud_stack -c acr_registry -e mlflow_experiment_tracker -a az_store -o k8s_orchestrator --model_deployer=seldon_deployer --set
} >> setup_out.log
