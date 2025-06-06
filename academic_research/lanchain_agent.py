from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import arxiv

# Load environment
load_dotenv()

# Define the tool function
@tool
def get_papers(query: str):
    """Fetch 3 relevant research papers from arXiv for a given query."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for result in client.results(search):
        results.append({
            "title": result.title,
            "summary": result.summary,
            "authors": [author.name for author in result.authors],
            "links": result.pdf_url
        })
    return results
tools = [get_papers]
# Initialize LLM
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")


prompt = PromptTemplate(
    input_variables=["query", "agent_scratchpad"],  # removed "tools"
    template="""
You are a helpful ReAct agent that uses the following tools:
[get_papers]

Your job is to find the top 3 relevant papers according to the user query: {query},
first understand the user query that what kind of papers user is looking for and make needful input from query to provide the tool 'get_papers'
note that you will only use the tool 'get_papers' to find the papers, you can't find by your self and tool finds paper from arxiv not other website.
The input to the tool will be a string query.
The tool will return a response in the form of a list of dictionaries.
Each dictionary includes: title, authors, summary, and links.
You must always return the response in the following format:

Format:

Paper 1:

Title:
Authors:
Summary:
Links:


Paper 2:

Title:
Authors:
Summary:
Links:


Paper 3:

Title:
Authors:
Summary:
Links:

your final response will be the format not any other things like what you think, what input you provide to tool, etc

{agent_scratchpad}
"""
)


agent = create_tool_calling_agent(
    llm=llm,
    tools=[get_papers],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[get_papers],
    # verbose=True,
    handle_parsing_errors=True
)

response = agent_executor.invoke({"query": "sentiment analysis"})
# print(response)

print(response['output'])
