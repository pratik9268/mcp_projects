import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
parser = StrOutputParser()

# Reinitialize embeddings (must match the one used to store vectors)
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    model_type="passage",
    api_key=nvidia_api_key
)

# Load existing vector store from disk
vector_store = Chroma(
    persist_directory='ecommerce_chroma_db',
    embedding_function=embeddings,
    collection_name='sample'
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")


# Step 1: Enhance the user's input query
enhancer_prompt = PromptTemplate.from_template("""
You are an intelligent query expander for product recommendations.

Original query: {user_query}

Rewrite the query with more detail, including possible product features, use cases, price range, and preferences.
""")


prompt = PromptTemplate(
    template="""
You are a knowledgeable e-commerce assistant helping users find ideal products.

User Preferences:
{user_query}

Product Catalog Entries:
{context}

Analyze the catalog and recommend 3-5 products that best suit the user's preferences.
For each recommended product, include:
-  Name
-  Key Features (2-3 concise bullets)
-  Why it's a good match

Focus on helpfulness, clarity, and accurate matching.
""",
    input_variables = ['context', 'user_query']
)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

enhacer_chain = enhancer_prompt | llm | parser

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'user_query': RunnablePassthrough()
})

main_chain = enhacer_chain | parallel_chain | prompt | llm | parser

i = input('query:')
answer = main_chain.invoke(i)
print(answer)
