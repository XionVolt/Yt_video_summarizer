# Streamlit-based YouTube Video Summarization App
import os
import subprocess
import re

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st

# ---------------------------- Setup ----------------------------
load_dotenv()

# ---------------------------- Helper Functions ----------------------------

def pull_model(model_name):
    st.write(f"Pulling `{model_name}` model...")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True, capture_output=True, text=True)
        st.success(f"Model `{model_name}` pulled successfully.")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to pull model: {e}")
        return False

def is_ollama_model_available(model_name: str) -> bool:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        return model_name in result.stdout
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to run 'ollama list': {e}")
        return False

def ready_model(model_name):
    key_prefix = f"{model_name}_status"

    # Initialize session state for the model
    if key_prefix not in st.session_state:
        st.session_state[key_prefix] = {
            "checked": False,
            "available": False,
            "declined": False,
            "model": None,
            "pull_requested": False,
        }

    state = st.session_state[key_prefix]

    # Check once if model is available
    if not state["checked"]:
        state["checked"] = True
        state["available"] = is_ollama_model_available(model_name)

    if state["available"]:
        if state["model"] is None:
            st.info(f"Model `{model_name}` is available and ready to use.")
            state["model"] = OllamaLLM(model=model_name)
        return state["model"]

    if state["declined"]:
        st.warning(f"You chose not to download `{model_name}`. Cannot proceed.")
        return None

    if state["pull_requested"]:
        with st.spinner(f"Pulling `{model_name}`..."):
            success = pull_model(model_name)
            if success:
                state["available"] = True
                state["model"] = OllamaLLM(model=model_name)
                st.success(f"Model `{model_name}` is now ready!")
                return state["model"]
            else:
                st.error(f"Failed to pull model `{model_name}`.")
                return None

    # Ask user to pull or decline
    st.warning(f"Model `{model_name}` is not available on your system.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Download Model", key=f"{model_name}_pull"):
            state["pull_requested"] = True
            st.rerun()
    with col2:
        if st.button("❌ No, Thanks", key=f"{model_name}_decline"):
            state["declined"] = True
            st.rerun()

    st.info("Please choose whether to download the model.")
    return None


def fetch_transcript(video_id):
    if not video_id:
        return None

    if ('https' in video_id) or ('youtu' in video_id):
        # Extract ID from full URL
        match = re.search(r"(?<=youtu.be/).*(?=\?)", video_id)
        if match:
            video_id = match.group()
            st.info(f"Extracted video ID: {video_id}")
        else:
            st.error("Invalid YouTube URL.")
            return None

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(chunk["text"] for chunk in transcript_list)
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

def generate_summary(transcript, model):
    prompt = PromptTemplate(
        template="""You are an assistant for summarizing YouTube videos. Use the given transcript to summarize this video.\nTranscript: {transcript}\n\nSummary: ...Generate Summary... """,
        input_variables=["transcript"]
    )
    final_prompt = prompt.format(transcript=transcript)
    answer = model.invoke(final_prompt)
    return answer

# ---------------------------- Streamlit UI ----------------------------

def app():
    st.title("🎬 YouTube Video Summarizer")

    # Sidebar inputs
    video_id = st.sidebar.text_input("Enter YouTube Video URL/ID", placeholder="ID or URL")
    modelOptions = st.sidebar.selectbox(
        "Choose model for generation",
        options=(
            "llama3.2", "mistral", "gemma:2b", "phi3:mini", "neural-chat", "llama2",
            "codellama", "openhermes", "dolphin-mistral", "qwen2.5-coder", "qwen3:0.6b"
        ),
    )

    if st.sidebar.button("Summarize Video"):
        if not video_id:
            st.warning("Please enter a YouTube video URL or ID.")
            return

        if not modelOptions:
            st.warning("Please select a model.")
            return

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

# ---------------------------- Entry ----------------------------

if __name__ == '__main__':
    app()
