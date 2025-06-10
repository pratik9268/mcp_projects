import streamlit as st
import asyncio
from fastmcp import Client

# Set up the page
st.set_page_config(page_title="Product Recommendation Assistant", page_icon="🛍️")
st.title("🛍️ Product Recommendation Assistant")
st.write("Describe what you're looking for and I'll find the best matches for you!")

# Initialize the client
client = Client("http://127.0.0.1:8000/mcp")

async def get_product_recommendations(user_query):
    async with client:
        return await client.call_tool("call_product_finder", {"usr_query": user_query})

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    st.info("This app connects to an AI-powered product recommendation engine.")

# Main chat interface
user_query = st.chat_input("What products are you looking for? (e.g. 'wireless headphones for running')")

if user_query:
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching for the best products..."):
            try:
                response = asyncio.run(get_product_recommendations(user_query))
                st.markdown(response[0].text)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Make sure the MCP server is running at http://127.0.0.1:8000")