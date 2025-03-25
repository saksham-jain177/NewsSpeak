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

class FinancialSentimentAnalyzer:
    def __init__(self):
        # Using FinBERT, specifically trained for financial text
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.labels = ["positive", "negative", "neutral"]

    def analyze_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get predicted label and confidence
        label_id = torch.argmax(predictions).item()
        confidence = predictions[0][label_id].item()
        
        return {
            "sentiment": self.labels[label_id],
            "confidence": confidence,
            "scores": {
                label: score.item() 
                for label, score in zip(self.labels, predictions[0])
            }
        }

class EnsembleSentimentAnalyzer:
    def __init__(self):
        self.finbert_analyzer = FinancialSentimentAnalyzer()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, text):
        # Get FinBERT analysis
        finbert_result = self.finbert_analyzer.analyze_sentiment(text)
        
        # Get VADER analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # Ensemble logic - weighing both models
        final_sentiment = self._combine_sentiments(
            finbert_result["sentiment"],
            vader_scores["compound"]
        )
        
        return {
            "sentiment": final_sentiment,
            "finbert_confidence": finbert_result["confidence"],
            "vader_score": vader_scores["compound"]
        }
    
    def _combine_sentiments(self, finbert_sentiment, vader_compound):
        # FinBERT is specifically trained for financial text
        finbert_weight = 0.7
        vader_weight = 0.3
        
        # If VADER shows strong sentiment, adjust weights
        if abs(vader_compound) > 0.5:
            vader_weight = 0.7
            finbert_weight = 0.3
        
        # Convert finbert sentiment to numeric
        finbert_numeric = {
            "positive": 1,
            "negative": -1,
            "neutral": 0
        }.get(finbert_sentiment, 0)
        
        # Weighted combination
        combined_score = (finbert_numeric * finbert_weight) + (vader_compound * vader_weight)
        
        # VADER's scale is -1 to 1, so adjust thresholds accordingly
        if combined_score <= -0.1:  # More strict negative threshold
            return "negative"
        elif combined_score >= 0.1:  # More strict positive threshold
            return "positive"
        return "neutral"

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
    """
    Enhanced comparative analysis with detailed insights
    """
    # Basic sentiment distribution
    sentiment_counts = Counter(sentiments)
    
    # Calculate percentages
    total = len(sentiments)
    sentiment_distribution = {
        sentiment: {
            'count': count,
            'percentage': (count/total) * 100
        } for sentiment, count in sentiment_counts.items()
    }
    
    # Trend analysis
    dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
    
    # Source diversity
    unique_sources = len(set(article['url'].split('/')[2] for article in articles if article['url']))
    
    return {
        "sentiment_distribution": sentiment_distribution,
        "dominant_sentiment": dominant_sentiment,
        "unique_sources": unique_sources,
        "total_articles": total,
        "confidence_score": (max(sentiment_counts.values()) / total) * 100
    }

# -------------------------------
# Function: generate_final_sentiment
# -------------------------------
def generate_final_sentiment(sentiments):
    """
    Determines the overall sentiment based on the majority vote from the list of sentiments.
    Returns the sentiment that occurs most frequently.
    """
    counts = Counter(sentiments)
    if counts:
        final_sentiment = counts.most_common(1)[0][0]
    else:
        final_sentiment = "Neutral"
    return final_sentiment

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

# Create a global instance of EnsembleSentimentAnalyzer
sentiment_analyzer = EnsembleSentimentAnalyzer()

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
                        "content": "You are a news summarizer. Create a one-sentence summary that provides context and implications beyond the headline. Focus on the key business or technical impact."
                    },
                    {
                        "role": "user",
                        "content": f"Based on this title and content, provide a meaningful one-sentence summary:\nTitle: {title}\nContent: {content}"
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
                        
                        Use bullet points (*) for each point. Be concise and specific."""
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
