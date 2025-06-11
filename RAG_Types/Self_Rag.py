import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
parser = StrOutputParser()

embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    model_type="passage",
    api_key=nvidia_api_key
)

vector_store = Chroma(
    persist_directory='DAA_notes_chroma_db',
    embedding_function=embeddings,
    collection_name='sample'
)

class Agentstste(TypedDict):
    question: str
    retrieved_docs: str
    grade_response: str
    answer: str
    hallucination_response: str
    check_answer_response: str

def retrive_docs(state: Agentstste):
    print('Node: Retrieve Docs')
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 30})
    docs = retriever.invoke(state["question"])
    state["retrieved_docs"] = "\n\n".join(doc.page_content for doc in docs)
    return state

def grade(state: Agentstste):
    print('\nNode: Grade Context')
    prompt = PromptTemplate(
        template="""You are a DAA expert. Decide if the retrieved context is relevant to the question.

        Question:
        {question}

        Context:
        {retrieved_docs}

        Respond ONLY with "yes" or "no".
        """, input_variables=['question', 'retrieved_docs']
    )
    chain = prompt | llm | parser
    result = chain.invoke({'question': state['question'], 'retrieved_docs': state['retrieved_docs']})
    state['grade_response'] = result
    return state

def grade_condition(state: Agentstste) -> str:
    return state["grade_response"]

def generate_answer(state: Agentstste):
    print('\n Node: Generate Answer')
    prompt = PromptTemplate(
        template="""You are a DAA professor. Use only the given context to answer the question.

        Question:
        {question}

        Lecture Notes Context:
        {retrieved_docs}

        Write clearly using examples, formulas, and structured explanation.
        """, input_variables=['question', 'retrieved_docs']
    )
    chain = prompt | llm | parser
    state['answer'] = chain.invoke({'question': state['question'], 'retrieved_docs': state['retrieved_docs']})
    return state

def check_hallucination(state: Agentstste):
    print('Node: Check Hallucination')
    prompt = PromptTemplate(
        template="""You are a DAA professor verifying if the answer is based only on the notes.

        Answer:
        {answer}

        Respond ONLY with "yes" (hallucinated) or "no" (faithful).
        """, input_variables=['answer']
    )
    chain = prompt | llm | parser
    result = chain.invoke({'answer': state['answer']})
    state['hallucination_response'] = result
    return state

def hallucination_condition(state: Agentstste) -> str:
    return state["hallucination_response"]

def chck_answers_question(state: Agentstste):
    print('Node: Check Answer Relevance')
    prompt = PromptTemplate(
        template="""Does the answer fully address the question in the DAA domain?

        Question:
        {question}

        Answer:
        {answer}

        Respond ONLY with "yes" or "no".
        """, input_variables=['answer', 'question']
    )
    chain = prompt | llm | parser
    result = chain.invoke({'answer': state['answer'], 'question': state['question']})
    state['check_answer_response'] = result
    return state

def check_answer_condition(state: Agentstste) -> str:
    return state["check_answer_response"]

def rewrite_question(state: Agentstste):
    print('Node: Rewrite Question')
    prompt = PromptTemplate(
        template="""Rewrite the following DAA question for clarity and better context matching:

Original:
{question}
""", input_variables=['question']
    )
    chain = prompt | llm | parser
    state['question'] = chain.invoke({'question': state['question']})
    return state

# Build LangGraph
graph = StateGraph(Agentstste)
graph.add_node("retrive_docs", retrive_docs)
graph.add_node("grade", grade)
graph.add_node("generate_answer", generate_answer)
graph.add_node("check_hallucination", check_hallucination)
graph.add_node("chck_answers_question", chck_answers_question)
graph.add_node("rewrite_question", rewrite_question)

graph.add_edge(START, 'retrive_docs')
graph.add_edge('retrive_docs', 'grade')
graph.add_conditional_edges("grade", grade_condition, {
    "yes": "generate_answer",
    "no": "rewrite_question"
})
graph.add_edge('generate_answer', 'check_hallucination')
graph.add_conditional_edges("check_hallucination", hallucination_condition, {
    "yes": "generate_answer",
    "no": "chck_answers_question"
})
graph.add_conditional_edges("chck_answers_question", check_answer_condition, {
    "yes": END,
    "no": "rewrite_question"
})
graph.add_edge("rewrite_question", "retrive_docs")

# Run
app = graph.compile()
initial_state1 = Agentstste(question='string matching')
response = app.invoke(initial_state1)

print("\nFinal Question:", response['question'])
print("\n Retrieved docs:",response['retrieved_docs'])
print("Final Answer:\n", response['answer'])
