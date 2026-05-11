import streamlit as st
from backend import ask
# Set the title for the Streamlit app
st.title("Boston Food Safety Intelligence Assistant 🍲 ")

# question input text box
question = st.text_input("Ask a question")


if st.button("Analyze"):
    result = ask(question, year=2025, k=5)

    st.write("Route:", result["route"])

    st.subheader("Answer")
    st.write(result["summary"])

    if result["type"] == "sql":
        st.subheader("Results")
        st.dataframe(result["data"])

        # call chart based on route

    elif result["type"] == "rag":
        st.subheader("Supporting Inspection Records")
        st.dataframe(result["data"])