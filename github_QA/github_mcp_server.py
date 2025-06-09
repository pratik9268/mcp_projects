import os
from fastmcp import FastMCP
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from git import Repo

load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

mcp = FastMCP("github_assistant_server")

# Clone the repository
repo_url = "https://github.com/patelvandan11/Movie-Recommendation-Model.git"
clone_dir = "cloned_repo"
Repo.clone_from(repo_url, clone_dir)

# Load and index the code
loader = DirectoryLoader(path=clone_dir, glob="**/*.py", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documents = loader.load()

splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=600, chunk_overlap=120)
chunks = splitter.split_documents(documents)

embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", model_type="passage", api_key=nvidia_api_key)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="github_chroma_db",
    collection_name="sample"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 40})

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

prompt = PromptTemplate(
    template="""You are an expert technical assistant for a GitHub repository...

Repository Context:
{context}

User Question: {question}
""",
    input_variables=["context", "question"]
)

parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | parser

@mcp.tool()
def call_github_qa(question: str):
    return main_chain.invoke(question)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
