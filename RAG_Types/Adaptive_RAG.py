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
    question: str
    retrieved_docs: str
    grade_response: str
    answer: str
    hallucination_response: str
    check_answer_response: str
    rewrite_count: int
    query_analysis_result :  str
    web_docs : str

def query_anlysis(state : Agentstste):
    print('\nNode: query analysis')
    prompt = PromptTemplate(
        template="""You are a routing question expert to vector stor or web search.

        The vector store contains information about DAA(Design and Analysis of Algorithms). its the notes of DAA concepts.

        now based on question you have to determine that retrive the docs from vector store or do web search.

        if you think that retrive the doc then response only with 'related to vectore store'.
        if you think to do web search then respond only with 'do web search'.

        Question:
        {question}

        Respond ONLY with "related to vectore store" or "no".
        """, input_variables=['question']
    )
    chain = prompt | llm | parser
    result = chain.invoke({'question': state['question']})
    state['query_analysis_result'] = result
    return state

def should_retriev_or_web_search(state : Agentstste):
    return state['query_analysis_result'].strip().lower()

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

def check_hallucination(state: Agentstste):
    print('Node: Check Hallucination')
    prompt = PromptTemplate(
        template="""You are a helpful assistance that checks if there is any hallucination in the answer.

        Answer:
        {answer}

        Respond ONLY with "yes" if there is hallucination or "no" if there is no hallucination.
        """, input_variables=['answer']
    )
    chain = prompt | llm | parser
    result = chain.invoke({'answer': state['answer']})
    state['hallucination_response'] = result
    return state

def hallucination_condition(state: Agentstste) -> str:
    return state["hallucination_response"].strip().lower()

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
    return state["check_answer_response"].strip().lower()

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

# web search node
def web_search(state : Agentstste):
    print('\n web search node')
    search = DuckDuckGoSearchRun()
    state['web_docs'] = search.invoke(state['question'])
    return state

def generate_using_web_docs(state : Agentstste):
    print('\n Node: Generate Answer using web docs')
    prompt = PromptTemplate(
        template="""You will generate response to the question. Use only the given web documents to answer the question.

        Question:
        {question}

        web search documents:
        {web_docs}

        """, input_variables=['question', 'retrieved_docs']
    )
    chain = prompt | llm | parser
    state['answer'] = chain.invoke({'question': state['question'], 'web_docs': state['web_docs']})
    return state

#graph building
graph = StateGraph(Agentstste)


graph.add_node("query_anlysis", query_anlysis)
graph.add_node("retrive_docs", retrive_docs)
graph.add_node("grade", grade)
graph.add_node("generate_answer_using_retrieved_docs", generate_using_retrieved_docs)
graph.add_node("check_hallucination", check_hallucination)
graph.add_node("chck_answers_question", chck_answers_question)
graph.add_node("rewrite_question", rewrite_question)
graph.add_node("web_search", web_search)
graph.add_node("generate_using_web_docs", generate_using_web_docs)


graph.add_edge(START, 'query_anlysis')
graph.add_conditional_edges("query_anlysis", should_retriev_or_web_search, {
    "related to vectore store": "retrive_docs",
    "do web search": "web_search"
})
graph.add_edge('retrive_docs', 'grade')
graph.add_conditional_edges("grade", grade_condition, {
    "yes": "generate_answer_using_retrieved_docs",
    "no": "rewrite_question"
})
graph.add_edge('generate_answer_using_retrieved_docs', 'check_hallucination')
graph.add_conditional_edges("check_hallucination", hallucination_condition, {
    "yes": "generate_answer_using_retrieved_docs",
    "no": "chck_answers_question"
})
graph.add_conditional_edges("chck_answers_question", check_answer_condition, {
    "yes": END,
    "no": "rewrite_question"
})
graph.add_edge("rewrite_question", "retrive_docs")
graph.add_edge("web_search", "generate_using_web_docs")
graph.add_edge("generate_using_web_docs", END)

app = graph.compile()

initial_state1 = Agentstste(question='dynamic Programming',rewrite_count=0)
response = app.invoke(initial_state1)

print("\n🔎 Final Question:", response['question'])
print("📘 Final Answer:\n", response['answer'])

# from IPython.display import Image, display
# display(Image(app.get_graph().draw_mermaid_png()))