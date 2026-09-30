import streamlit as st
st.title("welcomce to AI chatbot")
question=st.text_input("enter your question:")
if st.button("generate"):
    if question:
        st.success("you entered a question")
        st.write(question)
    else:
        st.error("please enter a question")