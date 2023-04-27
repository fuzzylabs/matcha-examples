import streamlit as st

st.title('LLM Summarization Demo')


def fetch_summary():
    ...


def main():
    input_text = st.text_area(label='Text to summarize', height=400)


    summarized_text = st.text_area(label="Summarized Text",
                                value="Dummy summary",
                                height=200
                                )
    

if __name__ == '__main__':
    main()