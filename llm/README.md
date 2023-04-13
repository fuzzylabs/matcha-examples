# LLM Summarization using `matcha`

In this example, we will fine-tune a Large Language Model (LLM) for summarization task. We will provision the resources required for running this example using `matcha`.

If you're wondering what on earth `matcha` is (besides the drink) then check out our main repository [here](https://github.com/fuzzylabs/matcha) and our [documentation](LINK) - don't forget to come back to try out this example!

## Getting Started

There's a bit of a setup required before unleashing `matcha`, the steps below will guide you through this.

Clone this repo:

```bash
git clone git@github.com:fuzzylabs/matcha-examples.git
```

Install matcha:

```bash
pip install matcha
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

Set up the environment:

This will install the requirements for the example (see [requirements.txt](requirements.txt)) and setup [ZenML](https://docs.zenml.io/getting-started/introduction):

```bash
./setup.sh
```

> You may need to give the `setup.sh` file the correct permissions to run, if so then do the following: `chmod +x setup.sh`.
>
> On completion, this will open a browser window for the ZenML server but you can ignore this for the time being (if you want to explore, then the username is `default` and there's no password - see the [ZenML documentation](https://docs.zenml.io/getting-started/introduction) for more information).

## `matcha` time

Use matcha to run the example 🍵:

```bash
matcha run
```

[Optional] Run the tests:

```bash
python -m pytest tests
```
