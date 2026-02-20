import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

st.set_page_config(page_title="Student Placement Assistant", page_icon="🎓")
st.title("🎓 Student Placement & Skill Chatbot")
st.write("Ask anything about student skills, placement, salary, performance, etc.")

# ============================
# Load Model + Prompt
# ============================

@st.cache_resource
def get_chain():
    model = OllamaLLM(model="gemma3:latest")

    template = """
    You are an expert Student Placement Assistant.

    Rules:
    - You MUST answer only using the provided student records.
    - Do NOT assume missing details.
    - If answer not present, say: "The data does not contain this information."

    Student Records:
    {records}

    User Question:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

chain = get_chain()

# ============================
# Chat History
# ============================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================
# Chat Input
# ============================

question = st.chat_input("Ask something like: 'Who has the highest CGPA?' or 'Which students got placed?'")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing student records..."):

            records = retriever.invoke(question)

            response = chain.invoke({
                "records": records,
                "question": question
            })

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})