import streamlit as st
import os
import asyncio
from fastmcp import Client
from langchain_community.document_loaders import PyPDFLoader

client = Client("http://127.0.0.1:8000/mcp")

# Title
st.title("📄 Resume Feedback and suggestion")

# File upload
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file
    pdf_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Document uploaded successfully!")

    # Load resume content
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    page_content = docs[0].page_content

    async def resume_agent(page_content):
        async with client:
            return await client.call_tool("call_analyze_resume", {"page_content": page_content})
    
    with st.spinner("Analyzing your resume..."):

        print("calling resume_agent")
        result = asyncio.run(resume_agent(page_content))
        print("get response from server")
        st.subheader("📝 Resume Feedback:")
        st.write(result[0].text
        
        )
    # Cleanup: remove the uploaded PDF file
    try:
        os.remove(pdf_path)
        print("Temporary file removed.")
    except Exception as e:
        print(f"Error removing temp file: {e}")