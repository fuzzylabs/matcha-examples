"""Implementation of the Huggingface data collator materializer."""

import os
from tempfile import TemporaryDirectory
from typing import Any, Type

from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.utils import io_utils

DEFAULT_TOKENIZER_DIR = "hf_tokenizer"


class HFTokenizerMaterializer(BaseMaterializer):
    """Materializer to read tokenizer to and from huggingface tokenizer."""

    ASSOCIATED_TYPES = (PreTrainedTokenizerBase,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: Type[Any]) -> PreTrainedTokenizerBase:
        """Reads Tokenizer.
        Args:
            data_type: The type of the tokenizer to read.
        Returns:
            The tokenizer read from the specified dir.
        """
        super().load(data_type)

        temp_dir = TemporaryDirectory()
        io_utils.copy_dir(
            os.path.join(self.uri, DEFAULT_TOKENIZER_DIR), temp_dir.name
        )

        return AutoTokenizer.from_pretrained(temp_dir.name)

    def save(self, tokenizer: Type[Any]) -> None:
        """Writes a Tokenizer to the specified dir.
        Args:
            tokenizer: The HFTokenizer to write.
        """
        super().save(tokenizer)
        temp_dir = TemporaryDirectory()
        tokenizer.save_pretrained(temp_dir.name)
        io_utils.copy_dir(
            temp_dir.name,
            os.path.join(self.uri, DEFAULT_TOKENIZER_DIR),
        )