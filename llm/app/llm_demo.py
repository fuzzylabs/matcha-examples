import streamlit as st
import json

st.title('LLM Summarization Demo')


def fetch_summary(txt: str) -> str:
    """Query seldon endpoint to fetch the summary.

    Args:
        txt (str): Input text to summarize

    Returns:
        str: Summarized text
    """
    ...


def read_examples(file_path: str = 'example.json') -> dict:
    """Read sample examples for LLM summarization demo.

    Args:
        file_path (str, optional): Path to json file. Defaults to 'example.json'.

    Returns:
        dict: Dictionary containing examples.
    """
    with open(file_path, 'r') as myfile:
        data = myfile.read()
    return json.loads(data)


def switch_examples(data: dict) -> str:
    """Switch between different examples.

    Args:
        data (dict): Dictionary containing examples.

    Returns:
        str: Input text to summarize.
    """
    pages = ["Example 1", "Example 2", "Example 3"]
    page = st.radio('Test Examples', pages)

    if page == "Example 1":
        text = data['example1']

    if page == "Example 2":
        text = data['example2']

    if page == "Example 3":
        text = data['example3']

    input_text = st.text_area(label='Text to summarize', value=text, height=400)
    return input_text


def main():

    data = read_examples(file_path="example.json")

    txt = switch_examples(data)

    result = st.button(label="Ready")

    if result:
        summarized_text = st.text_area(label="Summarized Text",
                                       value="Dummy summary",  # Replace with fetch_summary(txt)
                                       height=200
                                       )


if __name__ == '__main__':
    main()
