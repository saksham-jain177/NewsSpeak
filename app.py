import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

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
from utils import EnsembleSentimentAnalyzer, LLMSummarizer
import asyncio
import html
from typing import List, Dict
import re
import aiohttp

# Add after other imports
st.set_page_config(
    page_title="NewsSpeak",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated styling with optimized font loading and analytics blocking
st.markdown("""
    <style>
        /* Block Segment Analytics */
        script[src*='analytics.js'] { display: none !important; }
        
        /* Optimize font loading */
        @font-face {
            font-family: 'Source Sans Pro';
            font-style: normal;
            font-weight: 400;
            font-display: swap;  /* This helps with font loading */
            src: local('Source Sans Pro');
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
        
        /* Apply font globally */
        * {
            font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
        }
    </style>
    
    <!-- Completely disable Segment analytics -->
    <script>
        window.addEventListener('load', function() {
            // Disable analytics
            window.analytics = false;
            // Remove analytics script tag if it exists
            const analyticsScript = document.querySelector('script[src*="analytics.js"]');
            if (analyticsScript) analyticsScript.remove();
        });
    </script>
""", unsafe_allow_html=True)

# Download necessary NLTK data packages
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the ensemble sentiment analyzer
sentiment_analyzer = EnsembleSentimentAnalyzer()

# Initialize the LLM summarizer
llm_summarizer = LLMSummarizer()


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

def perform_comparative_analysis(sentiments):
    """
    Aggregate sentiment counts from a list of sentiment labels.
    """
    sentiment_counts = Counter(sentiments)
    return dict(sentiment_counts)

def generate_final_sentiment(sentiments):
    """
    Determine overall sentiment based on the majority sentiment.
    """
    counts = Counter(sentiments)
    if counts:
        final_sentiment = counts.most_common(1)[0][0]
    else:
        final_sentiment = "Neutral"
    return final_sentiment


# Module 5: Text-to-Speech Generation Function

async def translate_text(text):
    """Async function to translate text to Hindi"""
    translator = Translator()
    try:
        translation = await translator.translate(text, dest='hi')
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

async def generate_tts(text, filename="output.mp3"):
    """
    Translate the text to Hindi and generate an audio file using gTTS.
    Returns the filename of the generated audio.
    """
    try:
        # Directly await the translation
        hindi_text = await translate_text(text)
        
        if hindi_text is None:
            return None

        # Clean the text
        hindi_text = html.unescape(hindi_text)
        
        # Generate TTS
        tts = gTTS(text=hindi_text, lang='hi', slow=False)
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"TTS generation error: {str(e)}")
        return None


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

async def main():
    st.title("NewsSpeak: News Summarization & TTS Application")
    st.write("Enter a company name to fetch related news articles, perform sentiment analysis, and generate a Hindi audio summary.")

    # Add a status container at the top
    status_container = st.empty()

    company = st.text_input("Company Name", placeholder="e.g., Tesla")
    
    if st.button("Generate Report"):
        if not company:
            status_container.error("Please enter a company name.")
            return
        
        with st.spinner("Fetching articles..."):
            articles = scrape_articles(company)
        
        if not articles:
            status_container.error("No articles found. Try a different company name or check your connection.")
            return
        
        # Show source distribution
        sources = [article.get('source', 'Unknown') for article in articles]
        source_counts = Counter(sources)
        st.info(f"Found {len(articles)} articles")  # Simplified message

        # Initialize progress bar
        progress_bar = st.progress(0)
        
        # Initialize report structure
        report = {
            "Company": company,
            "Articles": [],
            "Comparative Sentiment Score": None,
            "Final Sentiment Analysis": None
        }
        
        sentiments = []
        
        for idx, article in enumerate(articles):
            # Ensure we're working with clean text
            clean_title = BeautifulSoup(article['title'], 'html.parser').get_text().strip()
            clean_content = BeautifulSoup(article.get('content', article.get('summary', '')), 'html.parser').get_text().strip()
            clean_content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', clean_content)
            clean_content = re.sub(r'\[.*?\]|\.\.\.|…', '', clean_content).strip()
            
            # Generate individual summary
            article_summary = await llm_summarizer.generate_article_summary(clean_title, clean_content)
            
            # Use the ensemble analyzer
            sentiment_result = sentiment_analyzer.analyze_sentiment(clean_content)
            sentiments.append(sentiment_result['sentiment'])
            topics = extract_topics(clean_content)
            
            article_data = {
                "Title": clean_title,
                "Summary": article_summary,  # Add the generated summary
                "Sentiment": sentiment_result['sentiment'],
                "FinBERT_Confidence": f"{sentiment_result['finbert_confidence']:.3f}",
                "VADER_Score": f"{sentiment_result['vader_score']:.3f}",
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
                "Time_Period": "Last 24 hours",  # You might want to make this dynamic
                "Sentiment_Distribution": {
                    "Positive": sentiments.count("positive"),
                    "Neutral": sentiments.count("neutral"),
                    "Negative": sentiments.count("negative")
                }
            }
        except Exception as e:
            st.warning("Could not generate overview summary. Continuing with detailed analysis.")
            report["Analysis_Summary"] = "Summary generation failed."
        
        # Comparative analysis
        comp_analysis = {
            "Sentiment Distribution": perform_comparative_analysis(sentiments),
            "Coverage Differences": "Detailed comparison not implemented",
            "Topic Overlap": "Detailed topic comparison not implemented"
        }
        report["Comparative Sentiment Score"] = comp_analysis
        final_sentiment = generate_final_sentiment(sentiments)
        
        # Enhanced final sentiment message
        confidence_scores = [float(article["FinBERT_Confidence"]) for article in report["Articles"]]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        report["Final Sentiment Analysis"] = (
            f"Overall sentiment for {company} is {final_sentiment} "
            f"(Average confidence: {avg_confidence:.2%})"
        )
        
        # Display detailed sentiment visualization first
        st.subheader("News Articles Analysis")
        for article in report["Articles"]:
            with st.expander(f"Article: {article['Title'][:100]}..."):
                st.write(f"Summary: {article['Summary']}")
                st.write(f"Sentiment: {article['Sentiment']}")
                st.write(f"FinBERT Confidence: {article['FinBERT_Confidence']}")
                st.write(f"VADER Score: {article['VADER_Score']}")
                st.write(f"Topics: {', '.join(article['Topics'])}")

        # Display overview summary after articles
        if report.get("Analysis_Summary"):
            st.subheader("Overall Analysis")
            st.write(report["Analysis_Summary"]["Overview"])
        
        # Display the structured report as JSON
        st.subheader("Structured Report")
        st.json(report)
        
        # Generate Hindi TTS for comprehensive summary
        st.info("Generating Hindi audio summary...")
        hindi_summary = await create_hindi_summary(report)
        tts_file = await generate_tts(hindi_summary)  # Add await here
        if tts_file and os.path.exists(tts_file):
            st.subheader("Hindi Audio Summary")
            st.audio(tts_file, format="audio/mp3")
        else:
            st.error("Failed to generate audio.")    

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
