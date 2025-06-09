import streamlit as st
import asyncio
from fastmcp import Client

# Initialize MCP client with your server URL
client = Client("http://127.0.0.1:8000/mcp")

st.title("📚 Library SQL Query Assistant")
st.write("Enter a natural language query to fetch data from the library database.")

# Input box for natural language query
nl_query = st.text_input("Your question about the library:")

async def sql_agent(nl_query):
    async with client:
        # Call the MCP tool with the key matching your mcp_server function
        return await client.call_tool("call_sql_query_perfomer", {"nl_query": nl_query})

if st.button("Run Query") and nl_query.strip():
    with st.spinner("Fetching results..."):
        response = asyncio.run(sql_agent(nl_query))
    # response is a list of results; display text of first result
    st.subheader("Results:")
    st.text(response[0].text if response else "No response from server.")
    print(response)
