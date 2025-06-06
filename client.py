import streamlit as st
import asyncio
from fastmcp import Client

# Initialize MCP client with your MCP server file path or URL
client = Client("mcp_server.py")

query = "sentiment analysis"

async def query_agent(query):
    async with client:
        return await client.call_tool("query_papers", {"query": query})

result = asyncio.run(query_agent(query))

print(type(result))
print(result[0].text)
# st.title("Arxiv Paper Finder")

# query = st.text_input("Enter your search query:")

# if st.button("Search"):
#     if query.strip():
#         with st.spinner("Fetching papers..."):
#             result = asyncio.run(query_agent(query))
#             st.markdown("### Results:")
#             st.markdown(result[0].text)  # format line breaks nicely
#     else:
#         st.warning("Please enter a query.")
