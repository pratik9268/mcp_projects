from langchain_community.document_loaders import PyPDFLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

pdf_path = 'pratik_resume.pdf'
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(type(docs[0].page_content))
docs

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

promt_template = PromptTemplate(
    template= """
    you are a helpful assistant which provides feedback and improvements to user resume.
    for that u you will get content of user's resume as {page_content}.
    your response will be improvements and feedback to user regarding their resume.
    In experinece section if address is added that will be the address of the company so note that point don't consider that address as candidate's.
    also don't mix candidate home address detail with candidates education address details.
    while providing the updated ressume based on suggestion and feedback do not change the fromat of resume. you can only suggest to change formate if you feel according to resume.
""",

input_variables=['page_content']
)

chain = promt_template | llm

response = chain.invoke({"page_content":docs[0].page_content})

print(response.content)