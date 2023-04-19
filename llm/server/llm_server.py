"""Server for LLM summarization fine-tuned model."""
import logging
import importlib
from typing import Optional, Any
import os
import numpy as np

from transformers import (
    AutoConfig,
    PreTrainedModel,
    AutoTokenizer,
    PreTrainedTokenizerBase
)

logger = logging.getLogger(__name__)
DEFAULT_PT_MODEL_DIR = "hf_pt_model"
DEFAULT_TOKENIZER_DIR = "hf_tokenizer"


class LLMServer(object):

    def __init__(
        self,
        model_uri: str = None,
        tokenizer_uri: str = None,
        working_directory: Optional[str] = None,
    ):
        """Initialise model server.

        Args:
            model_uri: URI pointing to the fine-tuned LLM model
            tokenizer_uri: URI pointing to tokenizer
            working_directory (Optional[str]): directory to download artifacts to, defaults to the directory of the script
        """
        super().__init__()

        if working_directory is None:
            working_directory = os.path.dirname(os.path.abspath(__file__))

        self.model_path = os.path.join(working_directory, DEFAULT_PT_MODEL_DIR)
        self.tokenizer_path = os.path.join(working_directory, DEFAULT_TOKENIZER_DIR)

        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.tokenizer_path, exist_ok=True)

        self.model = self.load_model(os.path.join(model_uri, DEFAULT_PT_MODEL_DIR))
        self.tokenizer = self.load_tokenizer(os.path.join(tokenizer_uri, DEFAULT_TOKENIZER_DIR))

    def load_model(self, model_uri: str) -> PreTrainedModel:
        """Load LLM model.

        Args:
            model_uri: Path where model is stored.

        Returns:
            PreTrainedModel: Loaded LLM model
        """
        # TODO: might have to implement a function to download from Azure Storage
        io_utils.copy_dir(model_uri, self.model_path)

        config = AutoConfig.from_pretrained(self.model_path)
        architecture = config.architectures[0]
        model_cls = getattr(
            importlib.import_module("transformers"), architecture
        )
        return model_cls.from_pretrained(self.model_path)

    def load_tokenizer(self, tokenizer_uri: str) -> PreTrainedTokenizerBase:
        """Load tokenizer.

        Args:
            tokenizer_uri: Path where tokenizer is stored.

        Returns:
            PreTrainedTokenizerBase: Loaded LLM tokemizer
        """
        # TODO: might have to implement a function to download from Azure Storage
        io_utils.copy_dir(tokenizer_uri, self.tokenizer_path)
        return AutoTokenizer.from_pretrained(self.tokenizer_path)

    def preprocess_input(self, input_text: str) -> np.ndarray:
        """Preprocess input text using tokenizer.

        Args:
            input_text (str): input text

        Returns:
        """
        input_id = self.tokenizer(input_text, return_tensors="pt").input_ids
        return input_id

    def predict(self, X: np.ndarray, feature_names: Optional[np.ndarray], **kwargs: Any):
        """Returns a prediction.

        Predict function in Seldon prediction function format to get predictions.

        Args:
            X (np.ndarray): input data
            feature_names (Optional[np.ndarray]): names of the features
            **kwargs: extra arguments

        Returns:.
        """
        try:
            input_id = self.preprocess_input(X["text"])
            outputs = self.model.generate(input_id,
                                          max_length=300,
                                          min_length=30,
                                          length_penalty=2.0,
                                          num_beams=4,
                                          no_repeat_ngram_size=3,
                                          early_stopping=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logging.exception("Exception during predict", e)
