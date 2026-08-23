# Conversational Chatbot with RAG (Memory-Aware)

A Streamlit chatbot that answers questions from an uploaded PDF and remembers prior turns in the conversation, using LangChain's history-aware retrieval chain.

## How it works
- Splits and embeds the PDF (`HuggingFace all-MiniLM-L6-v2` embeddings) into a FAISS vector store
- Uses `ChatGroq` as the LLM with `RunnableWithMessageHistory` to keep session-level chat memory
- Reformulates follow-up questions using conversation history before retrieving context

## Tech stack
Python · LangChain · Groq · HuggingFace Embeddings · FAISS · Streamlit

## Run locally
```bash
pip install -r requirements.txt
# set GROQ_API_KEY and HUGGING_FACE in a .env file
streamlit run app.py
```
