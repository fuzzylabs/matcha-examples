# 🎥🎞🍿 A movie recommendation example using `matcha` 

This is a straightforward movie recommendation example which can be run on the infrastructure provisioned by `matcha` 🍵. 

If you're wondering what on earth `matcha` is (besides the drink) then check out our main repository [here](https://github.com/fuzzylabs/matcha) and our [documentation](LINK) - don't forget to come back to try out this example!

## 🚦 Getting Started

There's a bit of a setup required before unleashing `matcha`, the steps below will guide you through this.

### 🧬 Cloning 

Before you start, this example workflow requires the Azure CLI to be installed. See [here](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) for how to do that.

You will also need to ensure you have installed [Docker](https://docs.docker.com/get-docker/) and that the Docker daemon is running on your machine. 

Clone this repo:

```bash
git clone git@github.com:fuzzylabs/matcha-examples.git
```

Go to the recommendation example directory:

```bash
cd matcha-examples/recommendation
```

Log into Azure via your terminal:

```bash
az login
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

> You need to be in the `recommendations` directory before running this!

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
Once setup.sh has completed, do the following to run the training pipeline:
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

## 🙋‍♀ Query the deployed model

✅ You've trained a model 

✅ You've deployed it 

❓ And now you want to get predictions. 


We've created a handy inference script which you can use to send a `user_id` and a `movie_id` to the deployed model get a predicted rating:

```bash
python inference.py --user 100 --movie 100
```

And the output should be something similar to:

```bash
User 100 is predicted to give the movie (100) a rating of: 4.2 out of 5
```

> Alternatively, you can `curl` the endpoint with the following:
> ```bash
> curl -XPOST -H 'Content-Type: application/json' -d '{"data": {"ndarray": [{"iid": "302", "uid": "196"}]}}' <endpoint_url>
> ```
>
> The output will be the raw predictions sent back by the model!

## Deprovision resources

Even though we've chosen a sensible default configuration for you, leaving the resources you've provisioned in this example running on Azure is going to run up a bill.

To deprovision the resources, run the following command:

```bash
matcha destroy
```