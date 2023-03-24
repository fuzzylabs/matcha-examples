#!/bin/bash
echo "Installing example requirements (see requirements.txt)..."
{
    pip install -r requirements.txt
    
    zenml integration install mlflow -y
} >> setup_out.log

mlflow_tracking_url=$(sed -n 's/.*"mlflow-tracking-url": "\(.*\)".*/\1/p' .matcha/infrastructure/matcha.state)

echo "Setting up ZenML (this will open a browser tab)..."
{
    export AUTO_OPEN_DASHBOARD=false
    zenml init 
    zenml up 
    zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow --tracking_uri="$mlflow_tracking_url" --tracking_username=username --tracking_password=password
    zenml stack register recommendation_example_stack -e mlflow_experiment_tracker -a default -o default --set
} >> setup_out.log