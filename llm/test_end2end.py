import os
import json
import requests
import pandas as pd
from functools import partial
from datasets import Dataset
from transformers import AutoTokenizer, BatchEncoding

DATASET_URL = "https://raw.githubusercontent.com/lauramanor/legal_summarization/master/all_v1.json"


def download_dataset(data_dir: str) -> None:
    data_path = os.path.join(data_dir, "summarization_dataset.json")
    if os.path.exists(data_path):
        print(f"Dataset already exists at {data_path}")
        json_data = open(data_path).read()
        data = json.loads(json_data)
        return data
    else:
        print("Downloading dataset")
        os.makedirs(data_dir, exist_ok=True)
        response = requests.get(DATASET_URL)
        data = response.json()
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Dataset downloaded to {data_dir}")
        return data


def convert_to_hg_dataset(data: dict):
    texts, summaries = [], []
    for _, v in data.items():
        texts.append(v["original_text"])
        summaries.append(v["reference_summary"])
    df = pd.DataFrame({"text": texts, "summary": summaries})
    dataset = Dataset.from_pandas(df)
    print("Max text length: ", max(df["text"].str.len()))
    print("Min text length: ", min(df["text"].str.len()))
    print("Max summary length: ", max(df["summary"].str.len()))
    print("Min summary length: ", min(df["summary"].str.len()))
    print(len(dataset), dataset)
    return dataset


def preprocess_function(dataset: Dataset, model_name, prefix):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = [prefix + doc for doc in dataset["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=dataset["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    print(isinstance(model_inputs, BatchEncoding))
    return model_inputs


def preprocess_dataset(dataset: Dataset, test_size: float, model_name: str, prefix: str):
    tokenized_data = dataset.map(partial(preprocess_function, model_name=model_name, prefix=prefix), batched=True)
    return tokenized_data.train_test_split(test_size=test_size)


if __name__ == "__main__":
    data_dir = "data/"
    prefix = "summarize: "
    model_name = "google/flan-t5-base"
    test_size = 0.2
    data_path = os.path.join(data_dir, "summarization_dataset.json")

    data = download_dataset(data_dir)
    dataset = convert_to_hg_dataset(data)
    tokenized_data = preprocess_dataset(dataset, test_size, model_name, prefix)
    print(tokenized_data)