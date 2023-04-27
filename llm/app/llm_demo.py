import streamlit as st

st.title('LLM Summarization Demo')


def fetch_summary():
    ...


def main():
    input_text = st.text_area(label='Text to summarize',
                    value='''Fuzzy Labs Limited (company no. 11762819) (“us”, “we”, or “our”) operates https://fuzzylabs.ai (the “Site”).
This page informs you of our policies regarding the collection, use and disclosure of Personal Information we receive from users of the Site.
We use your Personal Information only for providing and improving the Site.
By using the Site, you agree to the collection and use of information in accordance with this policy. Information Collection And Use
While using our Site, we may ask you to provide us with certain personally identifiable information that can be used to contact or identify you.
Personally identifiable information may include, but is not limited to your name and email address (“Personal Information”).
''',
        height=400
        )


    summarized_text = st.text_area(label="Summarized Text",
                                value="Dummy summary",
                                height=200
                                )
    

if __name__ == '__main__':
    main()