**AI Recipe Finder**

An intelligent recipe recommendation system that suggests food recipes based on user-provided ingredients. The system uses LangChain with NVIDIA AI models, Chroma vector database, and the MCP (Model Context Protocol) framework with a Streamlit-based frontend.


**Features**
 
🔍 Ingredient-based recipe search.

🧠 NVIDIA LLMs for recipe reasoning and formatting.

📚 Retrieval-augmented generation (RAG) using Chroma DB.

🌐 Streamlit UI for user-friendly interaction.

🔁 MCP-based server-client architecture for tool management.


**File	Description**

| File                               | Description                                                         |
| ---------------------------------- | ------------------------------------------------------------------- |
| `data_preprocess.ipynb`            | Preprocesses recipe data for embedding and storage in Chroma DB.    |
| `recipe_chatbot_langchain_code.py` | Basic CLI version of the recipe search using LangChain.             |
| `recipe_mcp_server.py`             | FastMCP server that serves a recipe search tool via WebSocket/HTTP. |
| `recipe_mcp_client.py`             | Streamlit frontend that interacts with the MCP server.              |
| `trial.ipynb`                      | Development notebook for intermediate testing.                      |
| `my_chroma_db/`                    | Local directory containing the embedded vector store.               |



**How It Works**

Embedding & Vector Store: Recipes are embedded using nvidia/nv-embedqa-e5-v5 and stored in Chroma.

LLM Prompting: A prompt template is used to instruct the LLM to format and return relevant recipes.

Retriever: The vector database retrieves top-matching recipes based on ingredient similarity.

Server (MCP): Hosts the tool as an MCP service via FastMCP.

Client (Streamlit): Users input ingredients, and the client fetches recipe results from the server.

