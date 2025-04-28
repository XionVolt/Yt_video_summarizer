# Streamlit-based YouTube Video Summarization App

import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

import streamlit as st

import re

# --------------------------------- Imports end here -----------------------------------
# Load environment variables
load_dotenv()

# --------------------------------- App starts here -----------------------------------
# Streamlit app configuration
st.set_page_config(layout="wide")
st.title("YouTube Video Summarization")
st.sidebar.title("Input")

# Sidebar input for YouTube video ID
video_id = st.sidebar.text_input("Enter YouTube Video URL/ID",placeholder="ID or URL")
st.sidebar.text("Note: This tool uses YouTube Transcript API and HuggingFace's LLM for summarization.")

# Function to fetch YouTube transcript
def fetch_transcript(video_id):
    # regex if user will paste video id
    if ('https' in video_id) or ('?' in video_id):
       video_id = re.search(r'(?<=youtu.be/).*(?=\?)', video_id).group()

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

# Function to split text into chunks
def split_text(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([transcript])

# Function to create embeddings and vector store
def create_vector_store(chunks):
    api_key = ''
    api_key = st.text_input("Enter your HuggingFace API key", type="password",key="huggingface_api_key")
    while api_key == '':
        pass
    else:
        st.success("API key set successfully.")

    # api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # if not api_key:
    st.error("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in your environment.")
        # return None
    embeddings = HuggingFaceInferenceAPIEmbeddings(model_name="intfloat/e5-base-v2", api_key=api_key)
    return FAISS.from_documents(chunks, embeddings)

# Function to retrieve context
def retrieve_context(vector_store, question):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever.invoke(question)

# Function to generate a summary
def generate_summary(context_text, question):
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context (retrieved from one YouTube video) to answer the question. If you don't know the answer, just say that you don't know.
        \nQuestion: {question}
        \nRetrieved Context: {context}""",
        input_variables=["question", "context"],
    )
    context = '\n\n'.join([doc.page_content for doc in context_text])
    final_prompt = prompt.invoke({'context': context, 'question': question})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=os.getenv('GOOGLE_API_KEY'))
    answer = llm.invoke(final_prompt)
    return answer.content

# Main function to run the app
def main():
    if st.sidebar.button("Summarize Video"):
        st.write("Fetching transcript...")
        transcript = fetch_transcript(video_id)
        if transcript:
            st.write("Splitting text into chunks...")
            chunks = split_text(transcript)
            st.write("Creating vector store...")
            vector_store = create_vector_store(chunks)
            if vector_store:
                st.write("Retrieving context...")
                question = "Summarize this video"
                context_text = retrieve_context(vector_store, question)
                if context_text:
                    st.write("Generating summary...")
                    summary = generate_summary(context_text, question)
                    st.subheader("Summary")
                    st.write(summary)

if __name__ == "__main__":
    main()