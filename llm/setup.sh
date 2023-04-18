#!/bin/bash
echo "Installing example requirements (see requirements.txt)..."
{
    pip install -r requirements.txt
    
    zenml integration install huggingface pytorch seldon kubernetes -y
} >> setup_out.log


echo "Setting up ZenML..."
{
    export AUTO_OPEN_DASHBOARD=false

    zenml init
    zenml stack register llm_example_stack -o default -a default --set

} >> setup_out.log
