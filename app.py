
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
from typing import List, Dict, Optional

# Define SUPPORTED_LANGUAGES at the module level
SUPPORTED_LANGUAGES = {
    'hi': 'Hindi',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh-cn': 'Chinese (Simplified)',
    'ar': 'Arabic',
    'ru': 'Russian'
}

# Load environment variables
load_dotenv()

import feedparser
import requests
from bs4 import BeautifulSoup
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
from collections import Counter
from gtts import gTTS
from googletrans import Translator
import os
from utils import SentimentAnalyzer, LLMSummarizer  # Updated import
import asyncio
import html
import re
import aiohttp
from datetime import datetime, timedelta
import json

# Add after other imports
st.set_page_config(
    page_title="NewsSpeak",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated styling with progress bar enhancement
st.markdown("""
    <style>
        /* Block Segment Analytics */
        script[src*='analytics.js'] { display: none !important; }
        
        /* Optimize font loading */
        @font-face {
            font-family: 'Source Sans Pro';
            font-style: normal;
            font-weight: 400;
            font-display: swap;
            src: local('Source Sans Pro');
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Enhanced Progress bar styling */
        .stProgress > div > div > div > div {
            background: linear-gradient(to right, #28a745, #34d058);
            border-radius: 10px;
            height: 20px;
        }
        
        .stProgress > div > div > div {
            background-color: #f0f2f5;
            border-radius: 10px;
            height: 20px;
        }
        
        /* Progress label styling with better contrast */
        .progress-label {
            position: relative;
            font-size: 14px;
            font-weight: 600;
            margin: 8px 0;
            padding: 5px 10px;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            color: var(--text-color, #262730);
            text-shadow: 0 1px 1px rgba(255, 255, 255, 0.2);
        }

        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .progress-label {
                color: #ffffff;
                text-shadow: 0 1px 1px rgba(0, 0, 0, 0.2);
            }
        }
        
        /* Progress percentage highlight */
        .progress-percentage {
            color: #28a745;
            font-weight: 700;
        }
        
        /* Apply font globally */
        * {
            font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# Download necessary NLTK data packages
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()  # Updated initialization

# Initialize the LLM summarizer
llm_summarizer = LLMSummarizer()  # Updated initialization


# Module 1: News Scraping Function

def scrape_articles(company):
    """
    Scrape news articles related to the company from Inshorts and Google News as fallback.
    """
    # Try Inshorts first
    url = f"https://inshorts.com/en/search?q={company}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    articles = []
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        news_cards = soup.find_all('div', class_='news-card')
        
        for card in news_cards:
            title_elem = card.find('span', itemprop='headline')
            title = title_elem.text.strip() if title_elem else "No Title"
            
            content_div = card.find('div', class_='news-card-content')
            summary_elem = content_div.find('div', itemprop='articleBody') if content_div else None
            summary = summary_elem.text.strip() if summary_elem else "No Summary Available"
            
            link_elem = card.find('a', href=True)
            article_url = link_elem['href'] if link_elem else ""
            
            articles.append({
                'title': title,
                'summary': summary,
                'url': article_url,
                'content': summary,
                'source': 'Inshorts'
            })
        
        if articles:
            return articles
            
    except Exception as e:
        st.warning(f"Inshorts scraping failed, trying Google News... ({str(e)})")
    
    # Fallback to Google News RSS
    try:
        import feedparser
        
        rss_url = f"https://news.google.com/rss/search?q={company}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        if feed.entries:
            for entry in feed.entries[:10]:
                articles.append({
                    'title': entry.title,
                    'summary': entry.description,
                    'url': entry.link,
                    'content': entry.description,
                    'source': 'Google News'
                })
                
    except Exception as e:
        st.error(f"Both news sources failed. Please check your internet connection. ({str(e)})")
        
    return articles


# Module 2: Sentiment Analysis Function

def analyze_sentiment(text):
    """
    Analyze sentiment using VADER.
    Returns sentiment category and compound score.
    """
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        sentiment = "Positive"
    elif score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, score


# Module 3: Topic Extraction Function

def extract_topics(text, num_topics=3):
    """
    Extract topics by performing a simple frequency analysis on the text.
    Removes stopwords and punctuation.
    """
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    # Remove punctuation and stopwords
    words = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    if not words:
        return []
    
    freq = Counter(words)
    most_common = freq.most_common(num_topics)
    topics = [word for word, count in most_common]
    return topics


# Module 4: Comparative Analysis Functions

def perform_comparative_analysis(sentiments, articles):
    """
    Aggregate sentiment counts from a list of sentiment labels and calculate average score.
    """
    sentiment_counts = Counter(sentiments)
    average_score = sum(float(article["Score"]) for article in articles) / len(articles)
    return {
        "sentiment_distribution": dict(sentiment_counts),
        "average_score": average_score
    }

def generate_final_sentiment(sentiments, sentiment_scores):
    """
    Determine overall sentiment based on the majority sentiment and confidence.
    """
    if not sentiments:
        return "Neutral", 0.0
    
    sentiment_counts = Counter(sentiments)
    most_common_sentiment = sentiment_counts.most_common(1)[0][0]
    
    # Calculate confidence
    total_articles = len(sentiments)
    majority_count = sentiment_counts[most_common_sentiment]
    confidence = majority_count / total_articles
    
    return most_common_sentiment, confidence

def generate_final_sentiment(sentiments, sentiment_scores):
    """
    Determine overall sentiment based on the majority sentiment and confidence.
    """
    if not sentiments:
        return "Neutral", 0.0
    
    sentiment_counts = Counter(sentiments)
    most_common_sentiment = sentiment_counts.most_common(1)[0][0]
    
    # Calculate confidence
    total_articles = len(sentiments)
    majority_count = sentiment_counts[most_common_sentiment]
    confidence = majority_count / total_articles
    
    return most_common_sentiment, confidence


# Module 5: Text-to-Speech Generation Function

async def translate_text(text: str, target_lang: str = None) -> str:
    """Async function to translate text to target language"""
    if target_lang is None:
        target_lang = st.session_state.target_language
        
    translator = Translator()
    try:
        translation = await translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

async def generate_tts(text: str, filename: str = "output.mp3") -> Optional[str]:
    """
    Translate and generate TTS in the selected language.
    Returns the filename of the generated audio.
    """
    try:
        target_lang = st.session_state.target_language
        translated_text = await translate_text(text, target_lang)
        
        if translated_text is None:
            return None

        # Clean the text
        translated_text = html.unescape(translated_text)
        
        # Generate TTS
        tts = gTTS(text=translated_text, lang=target_lang, slow=False)
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"TTS generation error: {str(e)}")
        return None

async def create_translated_summary(report: Dict) -> str:
    """Create a comprehensive summary in the selected language"""
    company = report["Company"]
    articles = report["Articles"]
    final_sentiment = report["Final Sentiment Analysis"]
    
    target_lang = st.session_state.target_language
    
    # Create summary template based on language
    if target_lang == 'hi':
        summary = f"{company} के बारे में समाचार विश्लेषण:\n\n"
    else:
        summary = f"News analysis for {company}:\n\n"
        
    # Translate the basic stats
    summary += await translate_text(
        f"Total articles found: {len(articles)}\n"
        f"Overall sentiment: {final_sentiment}\n\n"
    )
    
    for idx, article in enumerate(articles, 1):
        try:
            translated_title = await translate_text(article['Title'])
            translated_sentiment = await translate_text(f"Sentiment: {article['Sentiment']}")
            translated_topics = await translate_text(f"Topics: {', '.join(article['Topics'])}")
            
            summary += f"{idx}. {translated_title}\n"
            summary += f"{translated_sentiment}\n"
            summary += f"{translated_topics}\n\n"
        except Exception as e:
            st.warning(f"Translation failed for article {idx}: {str(e)}")
            continue
    
    return summary


# Main Streamlit Application

def deduplicate_articles(articles):
    """Remove duplicate articles based on title similarity"""
    seen_titles = set()
    unique_articles = []
    
    for article in articles:
        # Normalize title by removing source and common patterns
        normalized_title = ' '.join(article['title'].split('-')[0].lower().split())
        
        if normalized_title not in seen_titles:
            seen_titles.add(normalized_title)
            unique_articles.append(article)
    
    return unique_articles

async def create_hindi_summary(report):
    """Create a comprehensive Hindi summary of all articles"""
    company = report["Company"]
    articles = report["Articles"]
    final_sentiment = report["Final Sentiment Analysis"]
    
    translator = Translator()
    summary = f"{company} के बारे में समाचार विश्लेषण:\n\n"
    summary += f"कुल {len(articles)} समाचार लेख मिले।\n"
    summary += f"समग्र भावना: {final_sentiment}\n\n"
    
    for idx, article in enumerate(articles, 1):
        # Use clean title that's already processed
        try:
            translated_title = await translator.translate(article['Title'], dest='hi')
            title = translated_title.text
        except Exception:
            title = article['Title']
            
        summary += f"समाचार {idx}: {title}\n"
        summary += f"भावना: {article['Sentiment']}\n"
        summary += f"विषय: {', '.join(article['Topics'])}\n\n"
    
    return summary

def query_articles(articles: List[Dict], query: str) -> List[Dict]:
    """Search through articles based on query"""
    query = query.lower()
    results = []
    
    for article in articles:
        if (query in article['title'].lower() or 
            query in article['content'].lower() or 
            query in ' '.join(article['topics']).lower()):
            results.append(article)
    
    return results

def calculate_time_period(articles):
    """Calculate the time period covered by the articles"""
    dates = [article.get('published_date') for article in articles if article.get('published_date')]
    if not dates:
        return "Recent articles"
    
    dates = sorted(dates)
    oldest = min(dates)
    newest = max(dates)
    
    if oldest.date() == newest.date():
        return f"Articles from {oldest.strftime('%B %d, %Y')}"
    else:
        return f"Articles from {oldest.strftime('%B %d')} to {newest.strftime('%B %d, %Y')}"

if 'current_report' not in st.session_state:
    st.session_state.current_report = None
if 'current_company' not in st.session_state:
    st.session_state.current_company = None
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'target_language' not in st.session_state:
    st.session_state.target_language = 'hi'  # Default to Hindi

async def main():
    st.title("NewsSpeak: News Summarization & TTS Application")
    
    # Language selector in the sidebar - move this to the top
    with st.sidebar:
        st.subheader("Language Settings")
        selected_language = st.selectbox(
            "Select Output Language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: SUPPORTED_LANGUAGES[x],
            index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.target_language)
        )
        if selected_language != st.session_state.target_language:
            st.session_state.target_language = selected_language
            st.session_state.audio_file = None  # Clear previous audio when language changes
    
    # This will now update reactively whenever the language changes
    current_language = SUPPORTED_LANGUAGES[st.session_state.target_language]
    st.write(f"Enter a company name to fetch related news articles, perform sentiment analysis, and generate a {current_language} audio summary.")

    status_container = st.empty()
    
    # Use session state to maintain the company name
    company = st.text_input("Company Name", 
                           value=st.session_state.current_company if st.session_state.current_company else "",
                           placeholder="e.g., Tesla")

    if st.button("Generate Report"):
        if not company:
            status_container.error("Please enter a company name.")
            return
            
        st.session_state.current_company = company
        # Clear previous audio file
        st.session_state.audio_file = None
        
        with st.spinner("Fetching articles..."):
            articles = scrape_articles(company)
        
        if not articles:
            status_container.error("No articles found. Try a different company name or check your connection.")
            return
        
        # Show source distribution
        sources = [article.get('source', 'Unknown') for article in articles]
        source_counts = Counter(sources)
        st.info(f"Found {len(articles)} articles")  # Simplified message

        # Initialize progress tracking
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Initialize report structure
        report = {
            "Company": company,
            "Articles": [],
            "Comparative Sentiment Score": None,
            "Final Sentiment Analysis": None
        }
        
        sentiments = []
        
        for idx, article in enumerate(articles):
            # Calculate percentage
            progress_percentage = (idx + 1) / len(articles)
            percentage_text = f"{int(progress_percentage * 100)}%"
            
            # Update progress bar and text with enhanced formatting
            progress_bar.progress(progress_percentage)
            progress_text.markdown(f"""
                <div class="progress-label">
                    Processing article {idx + 1} of {len(articles)} 
                    <span class="progress-percentage">({percentage_text})</span>
                </div>
            """, unsafe_allow_html=True)

            # Ensure we're working with clean text
            clean_title = BeautifulSoup(article['title'], 'html.parser').get_text().strip()
            clean_content = BeautifulSoup(article.get('content', article.get('summary', '')), 'html.parser').get_text().strip()
            clean_content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', clean_content)
            clean_content = re.sub(r'\[.*?\]|\.\.\.|…', '', clean_content).strip()
            
            # Generate individual summary
            article_summary = await llm_summarizer.generate_article_summary(clean_title, clean_content)
            
            # Use the sentiment analyzer
            sentiment_result = sentiment_analyzer.analyze_sentiment(clean_content)
            sentiments.append(sentiment_result['sentiment'])
            topics = extract_topics(clean_content)
            
            article_data = {
                "Title": clean_title,
                "Summary": article_summary,
                "Sentiment": sentiment_result['sentiment'],
                "Score": f"{sentiment_result['score']:.3f}",
                "Details": {
                    "Positive": f"{sentiment_result['scores']['pos']:.3f}",
                    "Neutral": f"{sentiment_result['scores']['neu']:.3f}",
                    "Negative": f"{sentiment_result['scores']['neg']:.3f}"
                },
                "Topics": topics
            }
            report["Articles"].append(article_data)
            
            # Update progress
            progress_bar.progress((idx + 1) / len(articles))

        # Generate overview summary with structured format
        try:
            overview_summary = await llm_summarizer.generate_overview_summary(articles, company)
            # Parse and structure the summary
            summary_sections = {
                "Key Findings": [],
                "Market Impact": [],
                "Business Implications": [],
                "Recommendations": []
            }
            
            # Process the summary into structured sections
            current_section = "Key Findings"
            for line in overview_summary.split('\n'):
                line = line.strip()
                if line:
                    if "impact" in line.lower():
                        current_section = "Market Impact"
                    elif "implication" in line.lower():
                        current_section = "Business Implications"
                    elif "recommend" in line.lower():
                        current_section = "Recommendations"
                    elif line.startswith('*') or line.startswith('-'):
                        summary_sections[current_section].append(line.lstrip('*- '))

            report["Analysis_Summary"] = {
                "Overview": {
                    section: points for section, points in summary_sections.items() if points
                },
                "Article_Count": len(articles),
                "Time_Period": calculate_time_period(articles),
                "Sentiment_Distribution": {
                    "Positive": sentiments.count("Positive"),
                    "Neutral": sentiments.count("Neutral"),
                    "Negative": sentiments.count("Negative")
                }
            }
        except Exception as e:
            st.warning("Could not generate overview summary. Continuing with detailed analysis.")
            report["Analysis_Summary"] = "Summary generation failed."
        
        # Comparative analysis
        comp_analysis = perform_comparative_analysis(sentiments, report["Articles"])
        report["Comparative Sentiment Score"] = {
            "Sentiment Distribution": comp_analysis["sentiment_distribution"],
            "Average Score": f"{comp_analysis['average_score']:.3f}",
            "Coverage Differences": "Detailed comparison not implemented",
            "Topic Overlap": "Detailed topic comparison not implemented"
        }
        
        final_sentiment, confidence = generate_final_sentiment(sentiments, [float(article["Score"]) for article in report["Articles"]])
        report["Final Sentiment Analysis"] = (
            f"Overall sentiment for {company} is {final_sentiment} "
            f"(Average score: {comp_analysis['average_score']:.3f}, Confidence: {confidence:.1f}%)"
        )
        
        # Store the report in session state
        st.session_state.current_report = report

    # Display report if it exists in session state
    if st.session_state.current_report:
        report = st.session_state.current_report
        
        # Display detailed sentiment visualization first
        st.subheader("News Articles Analysis")
        for article in report["Articles"]:
            with st.expander(f"Article: {article['Title'][:100]}..."):
                st.write(f"Summary: {article['Summary']}")
                st.write(f"Sentiment: {article['Sentiment']}")
                st.write(f"VADER Score: {article['Score']}")
                st.write(f"Detailed Scores: {article['Details']}")
                st.write(f"Topics: {', '.join(article['Topics'])}")

        # Display overview summary after articles
        if report.get("Analysis_Summary"):
            st.subheader("Overall Analysis")
            
            # Display all sections of the analysis summary
            st.write("Article Count:", report["Analysis_Summary"]["Article_Count"])
            st.write("Time Period:", report["Analysis_Summary"]["Time_Period"])
            
            # Display sentiment distribution
            st.write("Sentiment Distribution:")
            st.write(report["Analysis_Summary"]["Sentiment_Distribution"])
            
            # Display the structured overview sections
            if report["Analysis_Summary"]["Overview"]:
                for section, points in report["Analysis_Summary"]["Overview"].items():
                    if points:  # Only display sections with content
                        st.write(f"\n{section}:")
                        for point in points:
                            st.write(f"• {point}")
        
        # Display the structured report as JSON
        st.subheader("Structured Report")
        st.json(report)

        # Create a container for download buttons with better alignment
        st.write("")  # Add some spacing
        button_container = st.container()
        
        # Use columns with a smaller ratio to bring buttons closer together
        with button_container:
            col1, col_spacer, col2 = st.columns([4, 1, 4])  # Adjust ratio for better spacing

            with col1:
                report_json = json.dumps(report, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download Full Report",
                    data=report_json.encode('utf-8'),
                    file_name=f"{st.session_state.current_company}_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="full_report_download",
                    use_container_width=True  # Make button use full column width
                )

            with col2:
                sentiment_report = {
                    "company": st.session_state.current_company,
                    "sentiment_distribution": report["Analysis_Summary"]["Sentiment_Distribution"],
                    "time_period": report["Analysis_Summary"]["Time_Period"]
                }
                sentiment_json = json.dumps(sentiment_report, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download Sentiment Analysis",
                    data=sentiment_json.encode('utf-8'),
                    file_name=f"{st.session_state.current_company}_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="sentiment_download",
                    use_container_width=True  # Make button use full column width
                )
        
        # Generate audio summary in the selected language
        if st.session_state.audio_file is None:
            st.info(f"Generating {SUPPORTED_LANGUAGES[st.session_state.target_language]} audio summary...")
            translated_summary = await create_translated_summary(report)
            tts_file = await generate_tts(translated_summary)
            if tts_file and os.path.exists(tts_file):
                st.session_state.audio_file = tts_file
            else:
                st.error("Failed to generate audio.")
        
        # Display audio if it exists
        if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
            st.subheader("Audio Summary")
            st.audio(st.session_state.audio_file, format="audio/mp3")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
