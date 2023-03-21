# A basic recommendation example

This is a simple recommendation example that can be run on the infrastructure provisioned by matcha 🍵.

## Getting Started

1. Clone this repo:
```bash
git clone git@github.com:fuzzylabs/matcha-examples.git
```

2. Go to the recommendation example directory:
```bash
cd recommendation
```

3. Create a new Python virtual environment and activate it. For Linux/MacOS users:
```bash
python3 -m venv exampleenv
source exampleenv/bin/activate
```

4. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Once that's complete, you need to setup ZenML:

```bash
zenml init
zenml up
zenml stack register recommendation_example_stack -a default -o default --set
```
## Running the example
From here, you can either run the recommendation example with matcha:
```bash
match run
```

Or the tests:
```bash
python -m pytest tests
```