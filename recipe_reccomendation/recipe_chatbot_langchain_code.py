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

# Reinitialize embeddings (must match the one used to store vectors)
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    model_type="passage",
    api_key=nvidia_api_key
)

# Load existing vector store from disk
vector_store = Chroma(
    persist_directory='my_chroma_db',
    embedding_function=embeddings,
    collection_name='sample'
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided context.
      Context is about food recipe which includes title, direction, ingridients and link to the website of that food recipe.
      title is the name of the food recipe.
      direction describes the steps or process of making that food.
      ingridents describes that what are the things required to make that food.

      user will provide only ingridents that they have and you need to give them top 3 or 5 food recipe from the context relevant to the user ingridents.
      In your response you will provide title of that recipe, ingridents to make that food, how to make it (direction).
      the direction you provide has to be step wise and named is as steps not direction.

      If the context is insufficient, just say you don't know.

      {context}
      Ingriidents: {ingridents}
    """,
    input_variables = ['context', 'ingridents']
)

parser = StrOutputParser()

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'ingridents': RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | parser

i = input('ingridents:')
answer = main_chain.invoke(i)
print(answer)


