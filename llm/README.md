# 📚🏛📖 LLM Summarization example using `matcha`

In this example, we'll show you how to use matcha to setup a default cloud environment on Azure and hook up a Large Language Model (LLM) for summarization pipeline to run on that environment.

If you're wondering what on earth `matcha` is (besides the drink) then check out our main repository [here](https://github.com/fuzzylabs/matcha) and our [documentation](https://fuzzylabs.github.io/matcha/) - don't forget to come back to try out this example!

## 🚦 Getting Started

There's a bit of a setup required before unleashing `matcha`, the steps below will guide you through this.

### &#9989; Pre-requisites

Before you start, this example workflow requires the Azure CLI to be installed. See [here](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) for how to do that.

You will also need to ensure you have installed [Docker](https://docs.docker.com/get-docker/) and that the Docker daemon is running on your machine.

Finally, you also need to ensure that you have [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) installed on your machine.

### 🧬 Cloning

Clone this repo:

```bash
git clone git@github.com:fuzzylabs/matcha-examples.git
```

Go to the llm example directory:

```bash
cd matcha-examples/llm
```

Create a virtual environment:

```bash
python3 -m venv venv 
source venv/bin/activate
```

> There is a requirement for the Python version being used to be 3.8+. We recommend making use of [pyenv](https://github.com/pyenv/pyenv) to manage your versions.

Install matcha:

```bash
pip install matcha-ml
```

### Provisioning

> You need to be in the `llm` directory before running this!

```bash
matcha provision
```

Once that's finished, crack on!

Set up the environment:

This will install the requirements for the example (see [requirements.txt](requirements.txt)) and setup [ZenML](https://docs.zenml.io/getting-started/introduction):

```bash
./setup.sh
```

> You may need to give the `setup.sh` file the correct permissions to run, if so then do the following: `chmod +x setup.sh`.

## ▶️ Running the example

Once `setup.sh` has completed, do the following to run the training pipeline:

```bash
python run.py --train
```

Once training has finished, we can deploy our trained model by doing the following:

```bash
python run.py --deploy
```

We can also run both training and deployment with one command:

```bash
python run.py --train --deploy
```

[Optional] Run the tests:

```bash
python -m pytest tests
```

## :raising_hand_woman: Query the deployed model

:white_check_mark: You've trained a model

:white_check_mark: You've deployed it

:question: And now you want to get predictions.

We have created a streamlit demo. To start the demo, run the following command:

```bash
streamlit run app/llm_demo.py
```

> Alternatively, you can `curl` the endpoint with the following:
>
> ```bash
> curl -XPOST -H 'Content-Type: application/json' -d '{"data": {"ndarray": [{"text": "Hi can you summarize this!"}]}}' <endpoint_url>
> ```
>
> The output will be the summary sent back by the model!

## Destroy resources

Even though we've chosen a sensible default configuration for you, leaving the resources you've provisioned in this example running on Azure is going to run up a bill.

To deprovision the resources, run the following command:

```bash
matcha destroy
```
