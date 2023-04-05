#!/bin/bash
echo "Installing example requirements (see requirements.txt)..."
{
    pip install -r requirements.txt
    
    zenml integration install mlflow seldon -y
} >> setup_out.log

if [[ ! -f .matcha/infrastructure/matcha.state ]]
then
    echo "Error: The file .matcha/infrastructure/matcha.state does not exist!"
    echo "Ensure that you have run 'matcha provision' in this directory and all cloud resources have been provisioned."
    exit 1
fi

mlflow_tracking_url=$(sed -n 's/.*"mlflow-tracking-url": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)
kubernetes_context=$(sed -n 's/.*"k8s-context": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)
seldon_workload_namespace=$(sed -n 's/.*"seldon-workloads-namespace": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)
seldon_ingress_host=$(sed -n 's/.*"seldon-base-url": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)

echo "Setting up ZenML (this will open a browser tab)..."
{
    export AUTO_OPEN_DASHBOARD=false
    zenml init 
    zenml up 
    zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow --tracking_uri="$mlflow_tracking_url" --tracking_username=username --tracking_password=password
    
    # Register the Seldon Core Model Deployer
    zenml model-deployer register seldon_deployer --flavor=seldon \
        --kubernetes_context=$kubernetes_context \
        --kubernetes_namespace=$seldon_workload_namespace \
        --base_url=http://$seldon_ingress_host \
        
    zenml stack register recommendation_example_stack -e mlflow_experiment_tracker -a default -o default --model_deployer=seldon_deployer --set
} >> setup_out.log
