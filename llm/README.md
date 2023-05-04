# 📚🏛📖 LLM Summarization example using `matcha`

In this example, we will fine-tune a Large Language Model (LLM) for summarization task. We will provision the resources required for running this example using `matcha` 🍵.

If you're wondering what on earth `matcha` is (besides the drink) then check out our main repository [here](https://github.com/fuzzylabs/matcha) and our [documentation](LINK) - don't forget to come back to try out this example!

## 🚦 Getting Started

There's a bit of a setup required before unleashing `matcha`, the steps below will guide you through this.

### 🧬 Cloning

Clone this repo:

```bash
git clone git@github.com:fuzzylabs/matcha-examples.git
```

Go to the recommendation example directory:

```bash
cd llm
```

Create a virtual environment:

```bash
python3 -m venv venv 
source venv/bin/activate
```

> There is a requirement for the Python version being used to be 3.8+. We recommend making use of [pyenv](https://github.com/pyenv/pyenv) to manage your versions.

Install matcha:

```bash
pip install matcha
```

### ⚠️ Provisioning

The rest of this how-to will not work unless you've provisioned some resources! If you've skipped over doing that (👀) and are part way through this how-to, then let's provision those resources needed for this example workflow:

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

## DeDeprovision resources

Even though we've chosen a sensible default configuration for you, leaving the resources you've provisioned in this example running on Azure is going to run up a bill.

To deprovision the resources, run the following command:

```bash
matcha destroy
```
