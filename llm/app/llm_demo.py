import streamlit as st

st.title('LLM Summarization Demo')


def fetch_summary(txt: str) -> str:
    """Query seldon endpoint to fetch the summary.

    Args:
        txt (str): Input text to summarize

    Returns:
        str: Summarized text
    """
    ...


def switch_examples() -> str:
    """Switch between different examples.

    Returns:
        str: Input text to summarize.
    """
    pages = ["Example 1", "Example 2", "Example 3"]
    page = st.radio('Test Examples', pages)

    if page == "Example 1":
        text = ""

    if page == "Example 2":
        text = ""

    if page == "Example 3":
        text = ""

    input_text = st.text_area(label='Text to summarize', value=text, height=400)
    return input_text


def main():
    txt = switch_examples()

    result = st.button(label="Ready")

    if result:
        summarized_text = st.text_area(label="Summarized Text",
                                       value="Dummy summary",  # Replace with fetch_summary(txt)
                                       height=200
                                       )


if __name__ == '__main__':
    main()
