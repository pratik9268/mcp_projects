 Resume Feedback and Suggestion
 
This project allows users to upload a PDF resume and get feedback and suggestions using an AI model.


📁 Project Files

streamlit_client.py: Frontend app built with Streamlit for uploading and analyzing resumes.

resume_analyzer_server.py: Backend server using FastMCP and LangChain to process resume content.

🚀 How to Run

1.Install dependencies:

pip install -r requirements.txt

2.Start backend server:

python resume_analyzer_server.py

3.Start Streamlit app:

streamlit run streamlit_client.py

⚙️ Requirements

Python 3.8+

Streamlit

FastMCP

LangChain

NVIDIA AI endpoints

python-dotenv
