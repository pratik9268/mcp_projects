import os
from fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA,NVIDIAEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

mcp = FastMCP('recipe_chatbot_server')

def recipe_finder(ingridents : str):
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

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 30})

    llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

    prompt = PromptTemplate(
        template="""
        You are a helpful culinary assistant that suggests recipes based on the ingredients a user has. 
    Your task is to recommend 3-5 relevant recipes from the provided context.

    ### Rules:
    1. Answer STRICTLY from the provided context only
    2. Context contains recipes with:
       - Title: Name of the recipe
       - Ingredients: Items needed to make the recipe
       - Directions: Preparation steps
       - Link: Source URL (do not include this in response)
    3. User will provide only ingredients they currently possess
    4. For each recommended recipe, provide:
       - Title
       - All required ingredients (not just what user has)
       - Step-by-step preparation instructions
    5. Format each recipe response as follows:
       Title: [recipe name]
       
       Ingredients:
       - [ingredient 1]
       - [ingredient 2]
       ...
       
       Steps:
       1. [step 1]
       2. [step 2]
       ...

    ### Important:
    - If no matching recipes exist in context, respond "I couldn't find matching recipes with your ingredients"
    - Never invent recipes not present in the context
    - Ingredients list must be complete for each recipe
    - Directions must be numbered steps, not paragraphs

    Context: {context}
    User's Ingredients: {ingridents}

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

    response = main_chain.invoke(ingridents)
    return response


@mcp.tool()
def call_recipe_finder(ingridents: str):
    print("calling recipe_finder")
    result = recipe_finder(ingridents)
    return result

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

