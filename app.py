import streamlit as st
import pandas as pd
import plotly.express as px
import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Set Streamlit config
st.set_page_config(page_title="CSV Chat App", layout="wide")

st.title("🧠 Query Your CSV with Natural Language")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df)

    # OpenAI API Key Input (from Streamlit Secrets or Env Vars)
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        llm = OpenAI(temperature=0)

        # Query box
        st.subheader("💬 Ask a question")
        question = st.text_input("Ask your question here:")

        if st.button("Get Answer") and question:
            try:
                with st.spinner("Thinking..."):

                    # Prompt template for safe GPT querying
                    prompt = PromptTemplate(
                        input_variables=["question", "df_head", "columns"],
                        template="""
You are a data analyst. You are given a dataset with the following columns:
{columns}

Here are the first few rows of the dataset:
{df_head}

Answer the following question:
{question}
"""
                    )

                    chain = LLMChain(llm=llm, prompt=prompt)

                    # Run GPT-based Q&A
                    response = chain.run({
                        "question": question,
                        "df_head": df.head(5).to_string(),
                        "columns": ", ".join(df.columns)
                    })

                st.success(response)
            except Exception as e:
                st.error(f"Error: {e}")

        # Plotly chart
        st.subheader("📊 Scatter Plot")
        x_axis = st.selectbox("X-axis", df.columns)
        y_axis = st.selectbox("Y-axis", df.columns)
        color_axis = st.selectbox("Color Grouping (Optional)", ["None"] + list(df.columns))

        if st.button("Plot"):
            if color_axis != "None":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_axis)
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please set the OpenAI API key in Streamlit secrets or environment.")
