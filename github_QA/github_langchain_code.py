from git import Repo
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import shutil
import stat
from git import Repo

load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

repo_url = "https://github.com/patelvandan11/Movie-Recommendation-Model.git"
clone_dir = "cloned_repo"

Repo.clone_from(repo_url, clone_dir)

loader = DirectoryLoader(
    path=clone_dir,
    glob="**/*.py",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=600,
    chunk_overlap=120,
)
chunks = splitter.split_documents(documents)

embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    model_type="passage",
    api_key=nvidia_api_key
)

vector_store = Chroma.from_documents(
    embedding = embeddings,
    documents= chunks,
    persist_directory='github_chroma_db',
    collection_name='sample'
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 40})

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

prompt = PromptTemplate(
    template="""
   You are an expert technical assistant for a GitHub repository. Follow these rules strictly:

1. Response Accuracy Protocol:
- Treat all technical terms and acronyms as undefined until verified in context
- Never assume functionality - only describe what exists in the code
- For unclear concepts: "The code shows X, but doesn't explicitly define Y"
- Support claims with file references: (path/to/file.ext)

2. Code Disclosure Rules:
[ONLY provide code when:]
- User uses explicit code request phrases:
  ✓ "Show me the code for..."
  ✓ "Display the implementation of..."
  ✓ Contains verbs: "code", "snippet", "implement"
  ✓ Asks for "actual code" or "source"

[NEVER provide code when:]
- Asking "how does..." without code request
- Conceptual questions
- Architecture overviews

3. When Providing Code:
```<language>
# path/to/file.ext
<code>
Precede with 1-sentence context

Follow with 1-sentence relevance explanation

Verification Protocol:
For technical terms like "MCP":

Check if defined in:

Constants files

Config files

Documentation

If undefined:
"The term 'MCP' appears in <files> but isn't formally defined"

General Responses:

Concise, bullet-pointed when possible

Offer code viewing when technical depth needed:
"Would seeing the implementation help?"

Repository Context:
{context}

User Question: {question}

Respond in markdown using this structure:
[Answer]
[Supporting Files]
[Optional Code]
    """,
    input_variables = ['context', 'question']
)

parser = StrOutputParser()

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | parser

chat_history = [] 
while True:
    question = input('query:')

    if question.lower().strip() == 'exit':
       break
    else:
        answer = main_chain.invoke(question)
        print(answer)


def force_remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Cleanup only the cloned repo
try:
    shutil.rmtree(clone_dir, onerror=force_remove_readonly)
    print("✅ Cleanup complete: Deleted cloned_repo.")
except Exception as e:
    print(f"⚠️ Cleanup error: {e}")