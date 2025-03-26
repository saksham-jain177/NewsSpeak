from flask import Flask, request, jsonify, send_file
import os
import asyncio
import aiohttp
from utils import (
    fetch_from_source,
    LLMSummarizer,
    SentimentAnalyzer,  # Updated import
    extract_topics,
    perform_comparative_analysis,
    generate_final_sentiment,
    generate_tts
)
from typing import Optional, List, Dict
from pydantic import BaseModel, validator

class ArticleRequest(BaseModel):
    company: str
    
    @validator('company')
    def validate_company(cls, v):
        if not v.strip():
            raise ValueError('Company name cannot be empty')
        if len(v) < 2:
            raise ValueError('Company name too short')
        return v

app = Flask(__name__)
sentiment_analyzer = SentimentAnalyzer()  # Updated initialization
llm_summarizer = LLMSummarizer()

@app.errorhandler(Exception)
def handle_error(error):
    return jsonify({
        "error": str(error),
        "type": error.__class__.__name__
    }), 500

@app.route('/api/scrape', methods=['GET'])
async def api_scrape():
    """
    Endpoint to scrape articles for a given company.
    Expects a query parameter 'company'.
    """
    company = request.args.get('company')
    if not company:
        return jsonify({"error": "Please provide a company name."}), 400

    async with aiohttp.ClientSession() as session:
        articles = await fetch_from_source(session, "google_news", company)
        
    if not articles:
        return jsonify({"error": "No articles found."}), 404

    return jsonify({"company": company, "articles": articles})

@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    """
    Endpoint to analyze sentiment using VADER.
    Expects a JSON payload with a 'text' field.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Please provide text for sentiment analysis."}), 400

    text = data['text']
    result = sentiment_analyzer.analyze_sentiment(text)
    
    return jsonify({
        "analysis": {
            "sentiment": result["sentiment"],
            "score": f"{result['score']:.3f}",
            "details": {
                "pos": f"{result['scores']['pos']:.3f}",
                "neu": f"{result['scores']['neu']:.3f}",
                "neg": f"{result['scores']['neg']:.3f}"
            }
        }
    })

@app.route('/api/report', methods=['GET'])
async def api_report():
    """
    Enhanced endpoint to generate a full report with LLM summaries.
    """
    company = request.args.get('company')
    if not company:
        return jsonify({"error": "Please provide a company name."}), 400

    async with aiohttp.ClientSession() as session:
        articles = await fetch_from_source(session, "google_news", company)
        
    if not articles:
        return jsonify({"error": "No articles found."}), 404

    # Limit to 10 articles for processing
    articles = articles[:10]
    report = {"Company": company, "Articles": [], "Analysis": {}}
    sentiments = []

    # Process each article
    for article in articles:
        # Get sentiment analysis
        sentiment_result = sentiment_analyzer.analyze_sentiment(article['raw_content'])
        sentiments.append(sentiment_result['sentiment'])
        
        # Extract topics
        topics = extract_topics(article['raw_content'])
        
        article_data = {
            "Title": article['title'],
            "Summary": article['summary'],
            "Sentiment": sentiment_result['sentiment'],
            "Score": sentiment_result['score'],
            "Topics": topics,
            "URL": article['url']
        }
        report["Articles"].append(article_data)

    # Generate overview summary
    overview_summary = await llm_summarizer.generate_overview_summary(articles, company)
    
    # Add analysis components
    comp_analysis = perform_comparative_analysis(sentiments, articles)
    report["Analysis"] = {
        "Overview": overview_summary,
        "Sentiment_Distribution": comp_analysis["sentiment_distribution"],
        "Dominant_Sentiment": comp_analysis["dominant_sentiment"],
        "Source_Diversity": f"Based on {comp_analysis['unique_sources']} unique sources",
        "Confidence": f"{comp_analysis['confidence_score']:.1f}%"
    }
    
    final_sentiment = generate_final_sentiment(sentiments)
    report["Final_Sentiment"] = {
        "Overall": final_sentiment,
        "Summary": f"Analysis of {company} based on {len(articles)} articles shows {final_sentiment.lower()} sentiment with {comp_analysis['confidence_score']:.1f}% confidence."
    }

    return jsonify(report)

@app.route('/api/tts', methods=['POST'])
def api_tts():
    """
    Endpoint to generate a Hindi TTS audio file from provided text.
    Expects a JSON payload with a 'text' field.
    Returns the generated MP3 file.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Please provide text for TTS conversion."}), 400

    text = data['text']
    filename = "output.mp3"
    result = generate_tts(text, filename)
    if result is None or not os.path.exists(filename):
        return jsonify({"error": "TTS generation failed."}), 500

    return send_file(filename, mimetype="audio/mp3", as_attachment=True)

@app.route('/analyze', methods=['POST'])
async def analyze():
    try:
        request_data = ArticleRequest(**request.json)
        company = request_data.company
        
        async with aiohttp.ClientSession() as session:
            articles = await fetch_from_source(session, "google_news", company)
            
        if not articles:
            return jsonify({"error": "No articles found"}), 404
        
        report = await process_articles_for_analysis(articles, company)
        return jsonify(report)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

async def process_articles_for_analysis(articles: List[Dict], company: str) -> Dict:
    """Helper function to process articles for analysis"""
    report = {
        "Company": company,
        "Articles": [],
        "Analysis": {},
        "Final_Sentiment": {}
    }
    
    sentiments = []
    
    for article in articles:
        sentiment_result = sentiment_analyzer.analyze_sentiment(article['raw_content'])
        sentiments.append(sentiment_result['sentiment'])
        topics = extract_topics(article['raw_content'])
        
        article_data = {
            "Title": article['title'],
            "Summary": article['summary'],
            "Sentiment": sentiment_result['sentiment'],
            "Score": sentiment_result['score'],
            "Topics": topics
        }
        report["Articles"].append(article_data)
    
    # Generate overview summary
    overview_summary = await llm_summarizer.generate_overview_summary(articles, company)
    
    comp_analysis = perform_comparative_analysis(sentiments, articles)
    report["Analysis"] = {
        "Overview": overview_summary,
        "Sentiment_Distribution": comp_analysis["sentiment_distribution"],
        "Dominant_Sentiment": comp_analysis["dominant_sentiment"],
        "Confidence": f"{comp_analysis['confidence_score']:.1f}%"
    }
    
    final_sentiment = generate_final_sentiment(sentiments)
    report["Final_Sentiment"] = {
        "Overall": final_sentiment,
        "Summary": f"Analysis of {company} based on {len(articles)} articles shows {final_sentiment.lower()} sentiment with {comp_analysis['confidence_score']:.1f}% confidence."
    }
    
    return report

if __name__ == '__main__':
    app.run(debug=True)
