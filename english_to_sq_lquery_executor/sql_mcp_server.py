from fastmcp import FastMCP
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import sqlite3

load_dotenv()

mcp = FastMCP('natural_language_to_sql_server')

def sql_query_perfomer(nl_query : str):

    prompt = PromptTemplate(
        input_variables=["nl_query"],
        template="""
    You are an assistant that converts English to SQL.
    Assume the SQLite table "library" with columns:
    (id,title, author, genre, year_published, rating, available)

    Convert the following question to a SQL query ONLY (no explanation):
    Question: {nl_query}
    """
    )

    llm = ChatNVIDIA()
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"nl_query": nl_query})
    sql_query = response.strip().split('\n')[0]

    connection = sqlite3.connect("library.db")
    cursor = connection.cursor()

    try:
        cursor.execute(sql_query)
        cursor.execute("SELECT * FROM library")
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"SQL Error: {e}")
        print(f"Problematic query: {sql_query}")
        rows = []


    result = "\n".join([str(row) for row in rows]) or "No results found."
    connection.close()

    return result


@mcp.tool()
def call_sql_query_perfomer(nl_query: str):
    print("calling sql_query_perfomer")
    result = sql_query_perfomer(nl_query)
    return result

if __name__ == "__main__":
    mcp.run(transport="streamable-http")