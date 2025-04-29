# Streamlit-based YouTube Video Summarization App

import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi


from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

import streamlit as st

import re

# --------------------------------- Imports end here -----------------------------------
# Load environment variables
load_dotenv()

# --------------------------------- App starts here -----------------------------------
# Sidebar input for YouTube video ID
video_id = st.sidebar.text_input("Enter YouTube Video URL/ID", placeholder="ID or URL")

# Sidebar input for HuggingFace API Key
modelOptions = st.sidebar.selectbox(
    "Choose model for generation",
    
    options = (
        ""
        "llama3.2".capitalize(),         # Meta's latest model, great reasoning
        "mistral".capitalize(),        # Fast and small, good summaries
        "gemma:2b".capitalize(),       # Google’s lightweight model
        "phi3:mini".capitalize(),      # Compact, efficient Microsoft model
        "neural-chat".capitalize(),    # Open model tuned for helpfulness
        "llama2".capitalize(),         # Still solid, larger memory needed
        "codellama".capitalize(),      # If coding support is needed
        "openhermes".capitalize(),     # Instruction-tuned version of Mistral
        "dolphin-mistral".capitalize(),
        "qwen2.5-coder".capitalize()   # For videos that are related to coding 
    ),
)



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


def ready_the_model():
    model = modelOptions
    
    

# Function to generate a summary
def generate_summary(transcript):
    prompt = PromptTemplate(
        template="""You are an assistant for summarizing YouTube videos. Use the given transcript to summarize this video. 
        \nTranscript: {transcript}""",
        input_variables=["transcript"]
    )
    final_prompt = prompt.invoke({"transcript": transcript})
    llm = modelOptions
    answer = llm.invoke(final_prompt)
    return answer.content

# Main function to run the app
def main():
    if st.sidebar.button("Summarize Video"):
        st.write("Fetching transcript...")
        transcript = fetch_transcript(video_id)
        if transcript:
            st.write("Generating summary...")
            summary = generate_summary(transcript)
            st.subheader("Summary")
            st.write(summary)

if __name__ == "__main__":
    main()