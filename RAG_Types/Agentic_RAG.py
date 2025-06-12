import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun


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
    agent_response : str
    question: str
    retrieved_docs: str
    grade_response: str
    answer: str
    rewrite_count: int


def should_respond_direct_or_retrieve_doc(state : Agentstste):
    print('\nNode: should_respond_direct_or_retrieve_doc')
    prompt = PromptTemplate(
        template="""You are a routing question expert to vector stor or direct respond.

        The vector store contains information about DAA(Design and Analysis of Algorithms). its the notes of DAA concepts.

        now based on question you have to determine that retrive the docs from vector store or dirct respond to the question.

        if you think that retrive the doc then response only with 'related to vectore store'.
        if you think to do web search then respond only with 'direct respond'.

        Question:
        {question}

        Respond ONLY with "related to vectore store" or "direct respond".
        """, input_variables=['question']
    )
    chain = prompt | llm | parser
    result = chain.invoke({'question': state['question']})
    state['agent_response'] = result

    if result == 'direct respond':
        state['answer'] = llm.invoke(state['question']).content
        print('\n direct response')

    return state


#roouting function
def should_respond_direct_or_retrieve_doc_condition(state : Agentstste):
     
    if state['agent_response'] == 'direct respond':
        return "responded directly"
    elif state['agent_response'] == 'related to vectore store':
        return "related to vectore store"


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

#routing function
def grade_condition(state: Agentstste) -> str:
    return state["grade_response"].strip().lower()

def generate_using_retrieved_docs(state: Agentstste):
    print('\n Node: Generate Answer using retrieved docs')
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

def rewrite_question(state: Agentstste):
    print('Node: Rewrite Question')
    state['rewrite_count'] += 1
    if state['rewrite_count'] > 3:
        raise Exception("Maximum rewrite attempts reached.")
    prompt = PromptTemplate(
        template="""Rewrite the following DAA question for clarity and better context matching:

        Original:
        {question}
        """, input_variables=['question']
    )
    chain = prompt | llm | parser
    state['question'] = chain.invoke({'question': state['question']})
    return state

#graph building
graph = StateGraph(Agentstste)

graph.add_node("should_respond_direct_or_retrieve_doc",should_respond_direct_or_retrieve_doc)
graph.add_node("retrive_docs", retrive_docs)
graph.add_node("grade", grade)
graph.add_node("generate_answer_using_retrieved_docs", generate_using_retrieved_docs)
graph.add_node("rewrite_question", rewrite_question)

graph.add_edge(START, 'should_respond_direct_or_retrieve_doc')
graph.add_conditional_edges("should_respond_direct_or_retrieve_doc", should_respond_direct_or_retrieve_doc_condition, {
    "related to vectore store": "retrive_docs",
    "responded directly": END
})
graph.add_edge('retrive_docs', 'grade')
graph.add_conditional_edges("grade", grade_condition, {
    "yes": "generate_answer_using_retrieved_docs",
    "no": "rewrite_question"
})
graph.add_edge('rewrite_question', 'should_respond_direct_or_retrieve_doc')
graph.add_edge("generate_answer_using_retrieved_docs", END)

app = graph.compile()

# from IPython.display import Image, display
# display(Image(app.get_graph().draw_mermaid_png()))

initial_state1 = Agentstste(question='hello',rewrite_count=0)
response = app.invoke(initial_state1)

print("\n🔎 Final Question:", response['question'])
print("📘 Final Answer:\n", response['answer'])