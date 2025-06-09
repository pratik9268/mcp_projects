import streamlit as st
import asyncio
import os
from fastmcp import Client

client = Client("http://127.0.0.1:8000/mcp")

st.set_page_config(page_title="GitHub Code Assistant", layout="wide")
st.title("🧠 GitHub Repository Code Assistant")

question = st.text_area("Ask your technical question about the repo:", height=200)

async def github_agent(q):
    async with client:
        return await client.call_tool("call_github_qa", {"question": q})

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing repository and answering..."):
            try:
                response = asyncio.run(github_agent(question))
                st.success("Answer retrieved!")
                st.markdown(response[0].text)
            except Exception as e:
                st.error(f"Error: {e}")
