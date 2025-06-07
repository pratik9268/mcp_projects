from fastmcp import FastMCP
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP('resume_analyzer_server')

def analyze_resume(page_content : str):
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

    promt_template = PromptTemplate(
        template= """
        you are a helpful assistant which provides feedback and improvements to user resume.
        for that u you will get content of user's resume as {page_content}.
        your response will be improvements and feedback to user regarding their resume.
        In experinece section if address is added that will be the address of the company so note that point don't consider that address as candidate's.
        also don't mix candidate home address detail with candidates education address details.
        while providing the updated ressume based on suggestion and feedback do not change the fromat of resume. you can only suggest to change formate if you feel according to resume.
        and also check that given {page_content} is a resume or not. if it is any other documents data then give response only like i can only analyze resume please upload resume and not give any feedback and suggestions.
    """,

    input_variables=['page_content']
    )

    chain = promt_template | llm
    response = chain.invoke({"page_content":page_content})
    print("response generated")
    return response.content

@mcp.tool()
def call_analyze_resume(page_content: str):
    """Use LangChain agent to query papers."""
    print("calling anlyze_resume")
    result = analyze_resume(page_content)
    return result

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    # mcp.run(transport="streamable-http", host="127.0.0.1", port=9000)
    # mcp.run(
    #     transport="sse",
    #     host="127.0.0.1",
    #     port=4200,
    #     log_level="debug",
    #     path="/my-custom-sse-path",
    # )