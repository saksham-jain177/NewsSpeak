# NewsSpeak: News Summarization & TTS Application

A Streamlit-based application that fetches news articles, performs sentiment analysis, and generates Hindi audio summaries.

## Features

- News article scraping from Google News
- Sentiment analysis using ensemble approach (VADER + FinBERT)
- Text summarization using LLM
- Hindi translation and Text-to-Speech conversion
- REST API endpoints for all major functionalities

## Setup

1. Clone the repository:

    ```bash
    git clone <your-repo-url>
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file with your API keys:

    ```bash
    OPENROUTER_API_KEY=your_key_here
    ```

4. Run the application:

    ```bash
    streamlit run app.py
    ```

## API Endpoints

- `/api/scrape`: GET - Fetch news articles
- `/api/sentiment`: POST - Analyze text sentiment
- `/api/tts`: POST - Generate Hindi audio
- `/analyze`: POST - Complete article analysis

## Testing

Run the test suite:

```bash
python test_functionality.py
```

## License

MIT License
