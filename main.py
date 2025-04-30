# Streamlit-based YouTube Video Summarization App
import os
import subprocess
import time
import re

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st

# --------------------------------- Setup -----------------------------------
load_dotenv()

def pull_model(model_name):
    subprocess.run(["ollama", "pull", model_name], check=True, capture_output=True, text=True)

def is_ollama_model_available(model_name: str) -> bool:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        return model_name in result.stdout
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to run 'ollama list': {e}")
        return False

def ready_model(model_name):
    if not is_ollama_model_available(model_name):
        try:
            pull_model(model_name)
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to pull model: {e}")
            return None
    return OllamaLLM(model=model_name)

# --------------------------------- Streamlit App -----------------------------------

# Sidebar inputs
video_id = st.sidebar.text_input("Enter YouTube Video URL/ID", placeholder="ID or URL")
modelOptions = st.sidebar.selectbox(
    "Choose model for generation",
    options=(
        "llama3.2",
        "mistral",
        "gemma:2b",
        "phi3:mini",
        "neural-chat",
        "llama2",
        "codellama",
        "openhermes",
        "dolphin-mistral",
        "qwen2.5-coder",
        "qwen3:0.6b"
    ),
)

# Fetch YouTube transcript
def fetch_transcript(video_id):
    if not video_id:
        return None
        
    if ('https' in video_id) or ('?' in video_id):
        match = re.search(r'(?<=youtu.be/).*(?=\?)', video_id)
        st.info(f"Extracted ({match.group()}) from {video_id}")
        if match:
            video_id = match.group()

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(chunk["text"] for chunk in transcript_list)
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

# Generate summary
def generate_summary(transcript, model):
    prompt = PromptTemplate(
        template="""You are an assistant for summarizing YouTube videos. Use the given transcript to summarize this video.\nTranscript: {transcript}\n\nSummary:""",
        input_variables=["transcript"]
    )
    final_prompt = prompt.format(transcript=transcript)
    answer = model.invoke(final_prompt)
    return answer

# Run app
def app():
    if st.sidebar.button("Summarize Video"):
        if not video_id:
            st.warning("Please enter a YouTube video URL or ID.")
        elif not modelOptions:
            st.warning("Please select a model.")
        else:
            with st.spinner("Fetching transcript..."):
                transcript = fetch_transcript(video_id)
            if transcript:
                with st.spinner("Preparing model..."):
                    model = ready_model(modelOptions)
                if model:
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(transcript, model)
                    st.subheader("Summary")
                    st.write(summary)


if __name__ == '__main__':
    app()
