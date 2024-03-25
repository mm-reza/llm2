import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db, llm
a = llm("today?")
date_part = ', '.join(a.split(', ')[1:])  # Join and then split

st.title("Question and Answer System Based on Google Palm LLM and Langchain (customized for Power BI) 🌱")

st.markdown(f'**This LLM may have the most up-to-date information available until {date_part}**')

# btn = st.button("update Knowledgebase")
# if btn:
#     create_vector_db()
# create_vector_db()

question = st.text_input("Question: ")
try:
    if question:
        chain = get_qa_chain()
        response = chain(question)

        st.header("Answer")
        st.write(response["result"])
except:
    st.write('** Something went wrong. Please try different question. **')





