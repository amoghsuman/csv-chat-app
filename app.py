import streamlit as st
import pandas as pd
import plotly.express as px
import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Page configuration
st.set_page_config(page_title="CSV Chat Assistant", layout="wide")

# Custom header
st.markdown(
    "<h1 style='text-align: center; color: #00ffcc;'>🧠 Query Your CSV with Natural Language</h1>",
    unsafe_allow_html=True
)
st.caption("Upload a CSV file, ask questions in natural language, and visualize insights instantly.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Data Preview
    with st.container():
        st.subheader("📄 Data Preview")
        st.dataframe(df, use_container_width=True)

    # Get API key from environment (set in Streamlit Cloud Secrets)
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        llm = OpenAI(temperature=0)

        # Question-answer section
        with st.expander("💬 Ask Questions About This Data", expanded=True):
            question = st.text_input("Type your question here")

            if st.button("Get Answer") and question:
                try:
                    with st.spinner("Thinking..."):

                        # Prompt template
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

                        response = chain.run({
                            "question": question,
                            "df_head": df.head(5).to_string(),
                            "columns": ", ".join(df.columns)
                        })

                    st.success(response)
                except Exception as e:
                    st.error(f"Error: {e}")

        # Scatter Plot
        with st.expander("📊 Visualize with Scatter Plot", expanded=False):
            col1, col2, col3 = st.columns(3)
            x_axis = col1.selectbox("X-axis", df.columns)
            y_axis = col2.selectbox("Y-axis", df.columns)
            color_axis = col3.selectbox("Color (Optional)", ["None"] + list(df.columns))

            if st.button("Plot"):
                try:
                    if color_axis != "None":
                        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_axis)
                    else:
                        fig = px.scatter(df, x=x_axis, y=y_axis)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Plotting error: {e}")
    else:
        st.warning("🔐 Please set the OpenAI API key in Streamlit Secrets.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>Made with ❤️ by Amogh Suman</p>",
    unsafe_allow_html=True
)
