# A basic recommendation example

This is a simple recommendation example that can be run on the infrastructure provisioned by matcha 🍵.

## Getting Started

1. Clone this repo:
```bash
git clone git@github.com:fuzzylabs/matcha-examples.git
```

2. Install matcha:
```bash
pip install matcha
```

2. Go to the recommendation example directory:
```bash
cd recommendation
```

4. Install the required dependencies (Assuming you are already in a virtual environment):
```bash
pip install -r requirements.txt
```

5. Once that's complete, you need to setup ZenML:

```bash
zenml init
zenml up
zenml stack register recommendation_example_stack -a default -o default --set
```
6. Run the example:
```bash
matcha run
```

7. [Optional] run tests:
```bash
python -m pytest tests
```