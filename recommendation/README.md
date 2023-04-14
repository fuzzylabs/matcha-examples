# A movie recommendation example using `matcha`

This is a straightforward movie recommendation example which can be run on the infrastructure provisioned by `matcha` 🍵.

If you're wondering what on earth `matcha` is (besides the drink) then check out our main repository [here](https://github.com/fuzzylabs/matcha) and our [documentation](LINK) - don't forget to come back to try out this example!

## Getting Started

There's a bit of a setup required before unleashing `matcha`, the steps below will guide you through this.

Clone this repo:

```bash
git clone git@github.com:fuzzylabs/matcha-examples.git
```

Go to the recommendation example directory:

```bash
cd recommendation
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

⚠️ Provisioning ⚠️

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
>
> On completion, this will open a browser window for the ZenML server but you can ignore this for the time being (if you want to explore, then the username is `default` and there's no password - see the [ZenML documentation](https://docs.zenml.io/getting-started/introduction) for more information).

## `matcha` time

Use matcha to run the example 🍵:

> This will both train the recommendation model and deploy it.

```bash
matcha run
```

[Optional] Run the tests:

```bash
python -m pytest tests
```

## Query the deployed model

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
>
> ```bash
> curl -XPOST -H 'Content-Type: application/json' -d '{"data": {"ndarray": [{"iid": "302", "uid": "196"}]}}' <endpoint_url>
> ```
>
> The output will be the raw predictions sent back by the model!
