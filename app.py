import streamlit as st
import pandas as pd
import plotly.express as px
import os
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

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

    # OpenAI API Key Input
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        llm = OpenAI(temperature=0)

        # Safer agent type (does not run code)
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type="zero-shot-react-description"  # Safe for public use
            handle_parsing_errors=True
        )

        # Query box
        st.subheader("💬 Ask a question")
        question = st.text_input("Ask your question here:")

        if st.button("Get Answer") and question:
            try:
                with st.spinner("Thinking..."):
                    response = agent.run(question)
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
