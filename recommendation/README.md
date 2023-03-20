# A basic recommendation example

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
pip install -r requirements.txt
```

Once that's complete, you need to setup ZenML:

```bash
zenml init
zenml up
zenml stack register recommendation_example_stack -a default -o default --set
```

From there, you can either run the pipeline itself:
```bash
python run.py
```

Or the tests:
```bash
python -m pytest tests
```