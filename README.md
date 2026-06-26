# RAG From Scratch

A Retrieval-Augmented Generation (RAG) system built in Python that allows users to ask questions about PDF documents.

## Features

* PDF text extraction using PyMuPDF
* Text chunking for efficient retrieval
* Semantic embeddings using Sentence Transformers
* Vector similarity search
* LLM-powered answer generation using Groq (Llama 3)
* Interactive question-answering interface

## Tech Stack

* Python
* Sentence Transformers
* FAISS
* PyMuPDF
* Groq API
* Llama 3

## Project Structure

```
rag-from-scratch/
├── src/
│   ├── main.py
│   ├── ingest.py
│   ├── embed.py
│   ├── retrieve.py
│   └── llm.py
├── data/
├── requirements.txt
├── .env
└── README.md
```

## Setup

1. Create a virtual environment

```bash
py -3.11 -m venv venv
```

2. Activate the environment

```bash
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Add your Groq API key in `.env`

```env
GROQ_API_KEY=your_api_key
```

5. Run the application

```bash
python src/main.py
```

## Example Questions

* Where did the candidate study?
* What is the candidate's CGPA?
* What skills does the candidate possess?

## Future Improvements

* Streamlit UI
* Multi-document support
* Chat history
* Persistent vector database
