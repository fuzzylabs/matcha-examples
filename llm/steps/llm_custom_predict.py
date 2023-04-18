from typing import Any, Dict, List, Union
import numpy as np


Array_Like = Union[np.ndarray, List[Any], str, bytes, Dict[str, Any]]


def custom_predict(
    model: Any,
    tokenizer: Any,
    request: Array_Like,
) -> Array_Like:
    """Custom Prediction function for LLM models.

    Request input is in the format: [{"text": "I am a text!"}, {"text": "I am another text!"}] 
    where each dictionary is a sample of text to be summarized.

    The custom predict function is the core of the custom deployment, the
    function is called by the custom deployment class defined for the serving
    tool. The current implementation requires the function to get the model
    loaded in the memory and a request with the data to predict.

    Args:
        model (Any): The model to use for prediction.
        tokenizer (Any): The tokenizer to use for prediction.
        request: The request response is an array-like format.

    Returns:
        The prediction in an array-like format. (e.g: np.ndarray,
        List[Any], str, bytes, Dict[str, Any])
    """
    inputs = []

    for instance in request:
        text = instance["text"]
        inputs = tokenizer(text, return_tensors="pt").input_ids
        outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        inputs.append(pred)

    return inputs
