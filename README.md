# RAG for Text

This project is a local Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Chroma, and Ollama. It has two steps:

1. Ingest PDF documents into the local Chroma database.
2. Ask questions against the ingested documents through the chatbot.

## 1. Where to place documents for ingestion

Put all PDF files you want to ingest into the `docs/` folder in the project root.

The ingestion script reads the folder path from `PDF_PATH` in the `.env` file. By default, this repository points `PDF_PATH` to the local `docs/` directory, so any PDFs placed there will be picked up by `ingestion.py`.

If you want to use another folder, update `PDF_PATH` to that folder path.

## 2. Set up the environment variables

Create or update a `.env` file in the project root with the document folder path:

```env
PDF_PATH="C:/Users/sansi/OneDrive/Desktop/RAG_for_Text/docs"
```

Notes:

- Use the full absolute path to the folder containing your PDF files.
- Make sure the path points to a folder, not a single PDF file.
- Keep the `.env` file in the project root so both scripts can load it with `load_dotenv()`.

## 3. Install Ollama and the local models

This project uses Ollama for both embeddings and chat generation.

### Install Ollama

1. Download and install Ollama from https://ollama.com.
2. Start the Ollama service.
3. Confirm it is running at `http://localhost:11434`.

### Pull the required models

Open a terminal and run:

```bash
ollama pull nomic-embed-text
ollama pull minicpm-v4.6
```

Model usage in this repo:

- `nomic-embed-text` is used in `ingestion.py` and `ChatRetrieval.py` to create embeddings.
- `minicpm-v4.6` is used in `ChatRetrieval.py` as the chat model.

If you want to use different Ollama models, update the model names in the Python files to match what you have installed locally.

## 4. Install Python dependencies

Create and activate a virtual environment, then install the requirements:

```bash
pip install -r requirements.txt
```

## 5. Build the vector database

After placing your PDFs in `docs/`, run the ingestion script:

```bash
python ingestion.py
```

This script:

- loads PDFs from the folder defined by `PDF_PATH`
- splits them into chunks
- creates embeddings with Ollama
- stores the result in `db/chroma_db`

Run ingestion again whenever you add, remove, or update documents.

## 6. Use the chatbot

Once the vector database has been created, start the chatbot with:

```bash
python ChatRetrieval.py
```

How to use it:

- Type your question at the prompt and press Enter.
- The chatbot retrieves the top 3 relevant chunks from Chroma.
- It then answers using only the retrieved document context.
- Type `exit` to close the chatbot.

## Project files

- `ingestion.py` builds the Chroma database from PDFs.
- `ChatRetrieval.py` runs the interactive chatbot.
- `docs/` stores the PDF documents to ingest.
- `db/chroma_db/` stores the generated local vector database.

## Troubleshooting

- If ingestion finds no documents, check that `PDF_PATH` points to the correct folder and that the folder contains PDF files.
- If Ollama errors out, make sure the Ollama app or service is running and the models have been pulled.
- If the chatbot cannot answer, re-run `python ingestion.py` after updating your documents.
