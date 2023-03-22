#!/bin/bash
echo "Installing example requirements (see requirements.txt)..."
{
    pip install -r requirements.txt
} >> setup_out.log

echo "Setting up ZenML (this will open a browser tab)..."
{
    zenml init 
    zenml up 
    zenml stack register recommendation_example_stack -a default -o default --set
} >> setup_out.log