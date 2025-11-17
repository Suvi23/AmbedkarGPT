# AmbedkarGPT
A Simple Retrieval-Augmented Generation RAG prototype for internship project.This prototype loads a short excerpt from Dr. B.R. Ambedkar’s “Annihilation of Caste”, converts it into embeddings, stores them in a local ChromaDB vector store, retrieves relevant text chunks based on user questions, and finally uses a local LLM (Ollama + Mistral 7B) to answer queries.
This project meets all the technical requirements specified in the assignment.

## What this repo contains
- AmbedkarGPT.ipynb — the Google collab notebook
- speech.txt — the provided Ambedkar excerpt 
- requirements.txt — Python dependencies
- README.md — this file

# Features

1)Pure local RAG pipeline (no API keys, no cloud services).

2)Uses LangChain to orchestrate the pipeline.

3)Uses HuggingFace (all-MiniLM-L6-v2) for embeddings.

4)Stores embeddings in ChromaDB (local persistent DB).

5)Answer generation via Ollama running Mistral 7B locally.

6)Fully offline once models are downloaded.

7Simple and clean command-line question-answering interface.

# Installation & Setup

Follow these steps to run the project locally.

1) Clone the Repository
   
git clone https://github.com/Suvi23/AmbedkarGPT.git
cd AmbedkarGPT

3) Create Virtual Environment

# macOS/Linux --->   
                    python3 -m venv venv
                    source venv/bin/activate

# Windows (PowerShell)------>    

                   python -m venv venv.\venv\Scripts\Activate.ps1

3) Install Python Dependencies ---->
   
                   pip install -r requirements.txt

4) Install Ollama (System Installation)

# macOS/Linux/WSL----->> 
             curl -fsSL https://ollama.ai/install.sh | sh

# Windows---->
            Install Ollama from: https://ollama.com/download


5) Download the Required LLM (Mistral 7B)----->> ollama pull mistral
 Verify:------> ollama list

You should see mistral in the list.


6) Run the Application

Once everything is installed:   python AmbedkarGPT.py  OR

# if you wish to run in google collab the direclty run the file -----> AmbedkarGPT.ipynb

You will see: AmbedkarGPT — ask anything about the provided speech. 

# Example question:

Question: What does Ambedkar say is the real remedy for caste?

# Example output:

Answer:
Ambedkar argues that the real remedy is to destroy the belief in the sanctity of the shastras...

