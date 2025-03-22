import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent

st.set_page_config(page_title="CSV Chat Assistant", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "datasets" not in st.session_state:
    st.session_state["datasets"] = []
if "suggested_questions" not in st.session_state:
    st.session_state["suggested_questions"] = []

# ─────────────────────────────────────────
# 🧠 App Title and Upload Area
# ─────────────────────────────────────────
st.markdown(
    "<h3 style='text-align: center; color: #00ffcc;'>🧠 Talk To Your CSV in Natural Language</h3>",
    unsafe_allow_html=True
)
st.caption("Upload multiple CSVs, switch between them, and analyze each dataset seamlessly.")

st.markdown("### 📤 Upload CSV(s)")
uploaded_files = st.file_uploader("Upload one or more CSVs", type="csv", accept_multiple_files=True)

for file in uploaded_files or []:
    if file.name not in [name for name, _ in st.session_state["datasets"]]:
        try:
            file.seek(0)
            df = pd.read_csv(file)
            st.session_state["datasets"].append((file.name, df))
        except Exception as e:
            st.warning(f"Failed to read {file.name}: {e}")

dataset_names = [name for name, _ in st.session_state["datasets"]]
selected_dataset_name = st.selectbox("📂 Select a Dataset", dataset_names) if dataset_names else None

# Load OpenAI API Key
api_key = st.secrets.get("OPENAI_API_KEY")

if selected_dataset_name:
    df = next(df for name, df in st.session_state["datasets"] if name == selected_dataset_name)

    # Sidebar – Plot Settings
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Plot Settings")
        x_axis = st.selectbox("X-axis", df.columns)
        y_axis = st.selectbox("Y-axis", df.columns)
        color_axis = st.selectbox("Color (Optional)", ["None"] + list(df.columns))

    # 📄 Data Preview
    st.subheader(f"📄 Data Preview: `{selected_dataset_name}`")
    st.dataframe(df, use_container_width=True)

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        llm = OpenAI(temperature=0)

        # Initialize agent
        if "agent" not in st.session_state or st.session_state.get("current_dataset") != selected_dataset_name:
            st.session_state["current_dataset"] = selected_dataset_name
            with st.spinner(f"Initializing AI Agent for `{selected_dataset_name}`..."):
                try:
                    st.session_state["agent"] = create_pandas_dataframe_agent(
                        llm=llm,
                        df=df,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True,
                        handle_parsing_errors=True,
                        allow_dangerous_code=True
                    )
                    st.success("✅ AI Agent initialized!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize AI Agent: {e}")

        agent = st.session_state["agent"]

        # 🧠 Summary
        st.markdown("### 🧠 Dataset Overview & Insights")

        df_info = io.StringIO()
        df.info(buf=df_info)
        info_str = df_info.getvalue()

        value_counts_dict = {
            col: df[col].value_counts().head(2).to_dict()
            for col in df.select_dtypes(include='object').columns
        }

        value_counts_str = "\n".join(
            f"{col}: {', '.join([f'{k} ({v})' for k, v in counts.items()])}"
            for col, counts in value_counts_dict.items()
        )

        summary_prompt = PromptTemplate(
            input_variables=["shape", "columns", "info", "describe", "value_counts"],
            template="""
You are a data analyst. Given the following dataset info, generate a clear and concise summary in markdown format using the following structure:

**🧾 Overview:** 1–2 lines about the data  
**🔍 Key Insights:** 3–5 bullet points highlighting trends or distributions  
**📌 Notable Stats:** 2–3 bullet points covering variance, skew, or standout numerical stats

Dataset shape: {shape}
Column names: {columns}
Info: {info}
Describe: {describe}
Value counts: {value_counts}
"""
        )

        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        dataset_summary = summary_chain.run({
            "shape": str(df.shape),
            "columns": ", ".join(df.columns),
            "info": info_str,
            "describe": df.describe().to_string(),
            "value_counts": value_counts_str
        })

        st.markdown(dataset_summary, unsafe_allow_html=True)

        # 🔍 Suggested Questions
        st.markdown("### 🤖 AI-Suggested Questions")

        question_prompt = PromptTemplate(
            input_variables=["columns"],
            template="""
You are a data scientist. Given the dataset with the following columns:
{columns}

Suggest 5 interesting questions a user might ask.
"""
        )
        q_chain = LLMChain(llm=llm, prompt=question_prompt)
        raw_questions = q_chain.run({"columns": ", ".join(df.columns)}).split("\n")
        cleaned_questions = [
            q.strip().lstrip("12345.:-•* ") for q in raw_questions if q.strip()
        ]
        st.session_state["suggested_questions"] = cleaned_questions

        selected_question = None
        if st.session_state["suggested_questions"]:
            selected_question = st.radio("Click a question to get an answer:", st.session_state["suggested_questions"])
        else:
            selected_question = None

        if st.button("Get AI Answer") and selected_question:
            with st.spinner("Thinking..."):
                try:
                    raw_output = agent.run(selected_question)

                    refine_prompt = PromptTemplate(
                        input_variables=["question", "raw_output"],
                        template="""
You are a helpful data analyst assistant. Summarize the answer to the question below in a concise, quantified, and markdown-friendly way.

**Question:** {question}  
**Agent Raw Output:** {raw_output}  

Final response format:
- Quantified values wherever applicable
- Reasoning used
- 4–6 lines max
- Markdown-friendly format
"""
                    )
                    refine_chain = LLMChain(llm=llm, prompt=refine_prompt)
                    final_answer = refine_chain.run({
                        "question": selected_question,
                        "raw_output": raw_output
                    })

                    st.success(final_answer)
                    st.session_state["chat_history"].append((selected_question, final_answer))
                except Exception as e:
                    st.error(f"Agent error: {e}")

        # 💬 Manual Questions
        with st.expander("💬 Ask Questions About This Data", expanded=True):
            user_question = st.text_input("Ask your question here:")
            if st.button("Get Answer") and user_question:
                with st.spinner("Thinking..."):
                    try:
                        raw_output = agent.run(user_question)

                        refine_prompt = PromptTemplate(
                            input_variables=["question", "raw_output"],
                            template="""
You are a helpful data analyst assistant. Summarize the answer to the question below in a concise, quantified, and markdown-friendly way.

**Question:** {question}  
**Agent Raw Output:** {raw_output}  

Final response format:
- Quantified values wherever applicable
- Reasoning used
- 4–6 lines max
- Markdown-friendly format
"""
                        )
                        refine_chain = LLMChain(llm=llm, prompt=refine_prompt)
                        final_answer = refine_chain.run({
                            "question": user_question,
                            "raw_output": raw_output
                        })

                        st.success(final_answer)
                        st.session_state["chat_history"].append((user_question, final_answer))
                    except Exception as e:
                        st.error(f"Agent error: {e}")

        # 📝 Chat History
        st.markdown("### 📝 Chat History")
        if st.session_state["chat_history"]:
            for q, a in st.session_state["chat_history"]:
                st.markdown(f"**🧠 You:** {q}")
                st.markdown(f"**🤖 Assistant:** {a}")
                st.markdown("---")
            if st.button("🧹 Clear Chat History"):
                st.session_state["chat_history"] = []
        else:
            st.info("Ask a question to start the conversation.")

    else:
        st.warning("🔐 Please add your OpenAI API key in Streamlit Cloud secrets.")

    # 📊 Scatter Plot
    with st.expander("📊 Visualize with Scatter Plot", expanded=False):
        if x_axis and y_axis:
            if st.button("Plot"):
                try:
                    fig = px.scatter(
                        df,
                        x=x_axis,
                        y=y_axis,
                        color=color_axis if color_axis != "None" else None
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Plotting error: {e}")
else:
    st.info("📁 Upload one or more CSV files to begin.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>Made with ❤️ by GT Bharat LLP</p>",
    unsafe_allow_html=True
)
