# YouTube Video Summarizer

This project provides a tool to summarize YouTube videos using langchain and ollama models. Basically we give model a transcript and it will generate a summary for us. 

## Features

- Fetches video transcripts using YouTube APIs.
- Summarizes transcripts using langchaing and ollama models.
- Supports currently only English videos(But will support other languages soon).
- Easy-to-use interface for summarization.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Yt_video_Summarizer.git
    cd Yt_video_Summarizer
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the streamlit app:
    ```bash
    python -m streamlit run main.py
    ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Acknowledgments

- [YouTube Data API](https://developers.google.com/youtube/v3)
- [Streamlit](https://streamlit.io/)
- Inspiration from video summarization tools