import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

os.environ['HUGGING_FACE'] = os.getenv("HUGGING_FACE")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')


from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.title("Conversational OpenSourse ChatBot")
st.write("Upload PDF and chat with your own content")

api = st.text_input("Enter the Groq API Key: ", type="password")
if api:
    llm = ChatGroq(model='Gemma2-9b-It', api_key=api)

    session_id = st.text_input("Session_ID:", value="Default_Session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploade_files = st.file_uploader("Choose a PDF file", type='pdf', accept_multiple_files=True)
    if uploade_files:
        documents = []
        for uploade_file in uploade_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploade_file.getvalue())
                filename = file.name

                docs = PyPDFLoader(temppdf).load()
                documents.extend(docs)

        splitted_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splitted_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextual_system_prompt = (
            "Given the chat History and latest user question"
            "which might reference in the chat history,"
            "formulate the standalone question which can be understood"
            "without the chat history. do not answer the question,"
            "just fomulate as it is if needed, otherwise return it as it is"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextual_system_prompt),
                MessagesPlaceholder('chat_history'),
                ("user", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)

        system_prompt = (
            "You are the assistent for question-answering task,"
            'use the following pieces of retrieved context to answer'
            "the question, i you don't find the answer just say i couldn't find the answer to that question"
            "\n\n"
            '{context}'
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key='input',
            history_messages_key="chat_history",
            output_messages_key='answer'
        )

        user_input = st.text_input("Enter the Question: ")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
            # st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            # st.write("Chat History:", session_history.messages)