import os
import requests
from typing import Optional, List, Dict
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
from collections import Counter
from gtts import gTTS
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from functools import lru_cache
import asyncio
import aiohttp
import feedparser
import re
import html
from urllib.parse import unquote

# Download necessary NLTK data packages if not already available
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Define news sources
news_sources = ["google_news", "inshorts"]  # Add your news sources here

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, text):
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # More strict thresholds for financial news
        if compound >= 0.15:  # Increased threshold
            sentiment = "Positive"
        elif compound <= -0.15:  # Decreased threshold
            sentiment = "Negative"
        else:
            # Consider word frequency for neutral cases
            if scores['neg'] > 0.15:  # If significant negative words present
                sentiment = "Negative"
            elif scores['pos'] > 0.15:  # If significant positive words present
                sentiment = "Positive"
            else:
                sentiment = "Neutral"
            
        return {
            "sentiment": sentiment,
            "score": compound,
            "scores": scores
        }

# Create a global instance of SentimentAnalyzer
sentiment_analyzer = SentimentAnalyzer()

# -------------------------------
# Function: scrape_articles
# -------------------------------
def scrape_articles(company: str) -> List[Dict]:
    """Scrape articles from various sources"""
    articles = []
    
    # Google News RSS Feed
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={company}&hl=en-US&gl=US&ceid=US:en")
    
    for entry in feed.entries[:5]:  # Limit to 5 articles
        article = {
            'title': entry.title,
            'summary': entry.description,  # This contains HTML
            'url': entry.link,
            'source': 'Google News'
        }
        # Clean the article immediately after creation
        cleaned_article = clean_google_news_data(article)
        articles.append(cleaned_article)
    
    return articles

def clean_google_news_data(article):
    """Clean Google News RSS data and generate LLM summary"""
    try:
        # 1. First clean the title
        if 'title' in article:
            soup = BeautifulSoup(article['title'], 'html.parser')
            clean_title = soup.get_text().strip()
            # Remove source name from title if present (after ' - ')
            clean_title = clean_title.split(' - ')[0]
            article['title'] = clean_title

        # 2. Clean the content/description
        if 'summary' in article:
            soup = BeautifulSoup(article['summary'], 'html.parser')
            full_text = soup.get_text()
            
            # Remove URLs
            clean_content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', full_text)
            # Remove any remaining [...] or ... references
            clean_content = re.sub(r'\[.*?\]|\.\.\.|…', '', clean_content)
            
            clean_content = clean_content.strip()
            
            # If content is empty or just matches title, use a default prompt for the LLM
            if not clean_content or clean_content == clean_title:
                # Create a prompt based on the title
                clean_content = f"Based on the title '{clean_title}', provide a likely summary of what this article discusses."
            
            # Always generate an LLM summary
            try:
                payload = {
                    "model": "google/gemma-3-12b-it:free",
                    "messages": [
                        {"role": "system", "content": "You are a news summarizer. Based on the title and any available content, provide a clear, factual, one-sentence summary that adds context beyond just the title. Focus on the implications and broader context."},
                        {"role": "user", "content": f"Title: {clean_title}\nContent: {clean_content}\n\nProvide a one-sentence summary that goes beyond just restating the title."}
                    ]
                }
                
                response = requests.post(
                    llm_summarizer.api_url,
                    json=payload,
                    headers=llm_summarizer.headers
                )
                response.raise_for_status()
                
                llm_summary = response.json()["choices"][0]["message"]["content"].strip()
                if llm_summary and not ('<' in llm_summary or '>' in llm_summary or 'http' in llm_summary.lower()):
                    article['summary'] = llm_summary
                else:
                    article['summary'] = f"Analysis of {clean_title}"
                
            except Exception as e:
                print(f"LLM Summary Generation Error: {str(e)}")
                article['summary'] = f"Analysis of {clean_title}"
        
        return article

    except Exception as e:
        print(f"Article cleaning error: {str(e)}")
        if 'summary' not in article:
            article['summary'] = f"Analysis of {article.get('title', 'article')}"
        return article

# -------------------------------
# Function: analyze_sentiment
# -------------------------------
def analyze_sentiment(text):
    """
    Enhanced sentiment analysis using VADER with improved sensitivity
    and consideration of financial/news context.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    # Get individual scores
    compound = scores['compound']
    
    # Proper VADER threshold interpretation
    if compound >= 0.1:  # More strict positive threshold
        sentiment = "Positive"
    elif compound <= -0.1:  # More strict negative threshold
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, compound

# -------------------------------
# Function: extract_topics
# -------------------------------
def extract_topics(text):
    """Extract main topics from text"""
    # Clean the text first
    text = re.sub(r'&nbsp;|nbsp|\'\'|\'', '', text)  # remove special chars
    text = html.unescape(text)
    
    # Rest of your topic extraction logic...
    words = word_tokenize(text.lower())
    words = [word for word in words 
            if word not in stopwords.words('english')
            and word not in string.punctuation
            and len(word) > 2  # Avoid short meaningless tokens
            and word.isalnum()]  # Only keep alphanumeric words
    
    word_freq = Counter(words)
    most_common = word_freq.most_common(5)
    topics = [word for word, count in most_common]
    return topics

# -------------------------------
# Function: perform_comparative_analysis
# -------------------------------
def perform_comparative_analysis(sentiments, articles):
    """Analyze sentiment distribution and calculate confidence"""
    sentiment_counts = Counter(sentiments)
    total = len(sentiments)
    
    # Calculate distribution percentages
    distribution = {
        "Positive": (sentiment_counts["Positive"] / total) * 100,
        "Neutral": (sentiment_counts["Neutral"] / total) * 100,
        "Negative": (sentiment_counts["Negative"] / total) * 100
    }
    
    # Find the dominant sentiment based on both count and strength
    scores = [float(article.get("Score", 0)) for article in articles]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Determine dominant sentiment using both frequency and score strength
    if abs(avg_score) < 0.05:  # Using VADER's threshold
        dominant = "Neutral"
    else:
        dominant = "Positive" if avg_score > 0 else "Negative"
    
    # Calculate confidence based on the strength of the average score
    confidence = min(abs(avg_score) * 100, 100)  # Convert to percentage, cap at 100
    
    return {
        "sentiment_distribution": distribution,
        "dominant_sentiment": dominant,
        "confidence_score": confidence,
        "average_score": avg_score,
        "unique_sources": len(set(article.get('source', '') for article in articles))
    }

# -------------------------------
# Function: generate_final_sentiment
# -------------------------------
def generate_final_sentiment(sentiments, scores=None):
    """Generate final sentiment based on both frequency and score strength"""
    if not scores:
        return max(Counter(sentiments).items(), key=lambda x: x[1])[0]
    
    avg_score = sum(scores) / len(scores)
    if abs(avg_score) < 0.05:
        return "Neutral"
    return "Positive" if avg_score > 0 else "Negative"

# -------------------------------
# Function: generate_tts
# -------------------------------
def generate_tts(text, filename="output.mp3"):
    """
    Translates the given text to Hindi and generates an audio file using gTTS.
    Returns the filename of the generated MP3 audio file.
    """
    try:
        translator = Translator()
        translated = translator.translate(text, dest='hi')
        hindi_text = translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        hindi_text = text  # Fallback to original text if translation fails
    
    try:
        tts = gTTS(text=hindi_text, lang='hi')
        tts.save(filename)
    except Exception as e:
        print(f"TTS generation error: {e}")
        return None
    return filename

@lru_cache(maxsize=100)
def cached_sentiment_analysis(text):
    return sentiment_analyzer.analyze_sentiment(text)

async def fetch_articles_async(company):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for source in news_sources:
            tasks.append(fetch_from_source(session, source, company))
        return await asyncio.gather(*tasks)

def clean_google_news_data(article):
    """Clean Google News RSS data and generate LLM summary"""
    try:
        # 1. First clean the title
        if 'title' in article:
            soup = BeautifulSoup(article['title'], 'html.parser')
            clean_title = soup.get_text().strip()
            # Remove source name from title if present (after ' - ')
            clean_title = clean_title.split(' - ')[0]
            article['title'] = clean_title

        # 2. Clean the content/description
        if 'summary' in article:
            soup = BeautifulSoup(article['summary'], 'html.parser')
            full_text = soup.get_text()
            
            # Remove URLs
            clean_content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', full_text)
            # Remove any remaining [...] or ... references
            clean_content = re.sub(r'\[.*?\]|\.\.\.|…', '', clean_content)
            # Remove the title if it appears at the start of the summary
            if clean_content.startswith(clean_title):
                clean_content = clean_content[len(clean_title):].strip()
            
            clean_content = clean_content.strip()
            
            # If content is empty or just matches title, try to get original description
            if not clean_content or clean_content == clean_title:
                original_desc = article.get('description', '')
                if original_desc:
                    soup_desc = BeautifulSoup(original_desc, 'html.parser')
                    clean_content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', soup_desc.get_text())
                    clean_content = re.sub(r'\[.*?\]|\.\.\.|…', '', clean_content).strip()
            
            # If still empty, use a placeholder
            if not clean_content or clean_content == clean_title:
                clean_content = "Full article details not available."
            
            article['summary'] = clean_content
            
            # 3. Generate a summary using LLM
            try:
                payload = {
                    "model": "google/gemma-3-12b-it:free",
                    "messages": [
                        {"role": "system", "content": "You are a news summarizer. Provide a clear, factual, one-sentence summary without URLs or references."},
                        {"role": "user", "content": f"Summarize this news in one sentence: Title: {clean_title}. Content: {clean_content}"}
                    ]
                }
                
                response = requests.post(
                    llm_summarizer.api_url,
                    json=payload,
                    headers=llm_summarizer.headers
                )
                response.raise_for_status()
                
                llm_summary = response.json()["choices"][0]["message"]["content"].strip()
                if llm_summary and not ('<' in llm_summary or '>' in llm_summary or 'http' in llm_summary.lower()):
                    article['summary'] = llm_summary
                
            except Exception as e:
                print(f"LLM Summary Generation Error: {str(e)}")
        
        return article

    except Exception as e:
        print(f"Article cleaning error: {str(e)}")
        # Don't fallback to title for summary in case of error
        if 'summary' not in article:
            article['summary'] = "Error processing article content."
        return article

async def fetch_from_source(session: aiohttp.ClientSession, source: str, company: str) -> List[Dict]:
    """Fetch articles from a specific news source"""
    if source == "google_news":
        url = f"https://news.google.com/rss/search?q={company}&hl=en-US&gl=US&ceid=US:en"
        async with session.get(url) as response:
            text = await response.text()
            feed = feedparser.parse(text)
            
            articles = []
            for entry in feed.entries[:5]:
                # Get article content
                article_content = await fetch_article_content(session, entry.link)
                
                # Create base article
                article = {
                    'title': entry.title.split(' - ')[0],  # Remove source from title
                    'url': entry.link,
                    'source': 'Google News',
                    'raw_content': article_content or entry.get('description', '')
                }
                
                # Generate summary
                article['summary'] = await llm_summarizer.generate_article_summary(
                    article['title'],
                    article['raw_content']
                )
                
                articles.append(article)
            
            return articles
    return []

async def fetch_article_content(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Fetch and extract article content"""
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Try multiple selectors to find content
                for selector in ['article', 'main', '.article-content', '[itemprop="articleBody"]']:
                    if content := soup.select_one(selector):
                        return ' '.join(p.get_text(strip=True) for p in content.find_all('p'))
                
                # Fallback to first few paragraphs
                paragraphs = soup.find_all('p')
                return ' '.join(p.get_text(strip=True) for p in paragraphs[:5])
    except Exception as e:
        print(f"Error fetching article content: {str(e)}")
    return None

class LLMSummarizer:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8501",
            "Content-Type": "application/json"
        }

    async def generate_article_summary(self, title: str, content: str) -> str:
        """Generate a single article summary"""
        try:
            payload = {
                "model": "google/gemma-3-12b-it:free",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a news summarizer that creates concise, factual summaries.
                        STRICT RULES:
                        1. Generate ONLY ONE direct sentence based on the title
                        2. NO prefixes (like 'Summary:', 'Analysis:', etc.)
                        3. NO formatting markers or special characters
                        4. NO asking for context or providing options
                        5. Focus on business/technical implications
                        6. Start directly with the main point
                        7. If title is unclear, make a neutral statement based on available facts
                        8. Maximum length: 200 characters
                        
                        FORBIDDEN:
                        - Multiple sentences
                        - Bullet points
                        - Prefixes or labels
                        - Questions
                        - Formatting characters (_,*,etc)"""
                    },
                    {
                        "role": "user",
                        "content": f"Title: {title}"
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        summary = data["choices"][0]["message"]["content"].strip()
                        # Remove any potential prefixes or special characters
                        summary = summary.lstrip('_:*.- \n')
                        if summary and not ('<' in summary or '>' in summary or 'http' in summary.lower()):
                            return summary
            
            return f"Analysis of {title}"
        except Exception as e:
            print(f"Summary generation error: {str(e)}")
            return f"Analysis of {title}"

    async def generate_overview_summary(self, articles: List[Dict], company: str) -> str:
        try:
            context = f"Articles about {company}:\n\n"
            for idx, article in enumerate(articles[:5], 1):
                context += f"{idx}. {article['title']}\n"

            payload = {
                "model": "google/gemma-3-12b-it:free",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a professional business analyst providing structured insights from news articles.
                        Format your analysis in the following sections:
                        1. Key Findings (3-4 bullet points)
                        2. Market Impact (2-3 bullet points)
                        3. Business Implications (2-3 bullet points)
                        4. Recommendations (if applicable)
                        
                        Use simple bullet points without any special characters or formatting.
                        Each point should start directly with the main message.
                        Do not use asterisks, colons, or other formatting markers."""
                    },
                    {
                        "role": "user",
                        "content": f"{context}\nProvide a structured analysis of these articles following the specified format."
                    }
                ]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]

            return None
        except Exception as e:
            print(f"Overview summary generation error: {str(e)}")
            return None

# Add to existing sentiment_analyzer instance
llm_summarizer = LLMSummarizer()
