import streamlit as st
import os
import asyncio
from fastmcp import Client

client = Client("http://127.0.0.1:8000/mcp")

st.set_page_config(page_title="Recipe Chatbot", page_icon="🍽️", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 3rem;
    }
    .subtitle {
        font-size: 28px;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    /* Hide default label */
    label[for="ingredients"] {
        display: none;
    }
    .custom-label {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #333;
    }
    .stTextArea>div>textarea {
        font-size: 18px;
        min-height: 200px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍳 AI Recipe Finder")

st.markdown('<div class="subtitle">Enter ingredients you have, and get recipes you can cook!</div>', unsafe_allow_html=True)

# Custom big heading for text area
st.markdown('<div class="custom-label">Ingredients (comma-separated):</div>', unsafe_allow_html=True)

# Now use empty label to avoid default text
ingredients = st.text_area("", height=200, key="ingredients")

async def recipe_agent(ingredients):
    async with client:
        return await client.call_tool("call_recipe_finder", {"ingridents": ingredients})

if st.button("Find Recipes"):
    if not ingredients.strip():
        st.warning("Please enter some ingredients.")
    else:
        with st.spinner("Finding recipes..."):
            try:
                response = asyncio.run(recipe_agent(ingredients))
                st.success("Recipes found!")
                
                st.markdown(response[0].text)

                
            except Exception as e:
                st.error(f"Error: {str(e)}")
