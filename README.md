Similarity Tool
Overview

A Gradio-based app that helps researchers find similar clinical trials based on study description, design, inclusion, and exclusion criteria. It uses: SentenceTransformers + Pinecone for retrieval LLMs (GPT, Gemini, Llama, Deepseek) for reasoning and exclusions A simple Gradio UI Setup Clone repo & enter folder :

git clone https://github.com/<USERNAME>/<REPO>.git
cd <REPO>


Create virtual environment

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux


Install dependencies

pip install -r requirements.txt


Add API keys (OpenAI, Pinecone, Groq, Google GenAI). Use env vars or .env.

Run app

python app.py