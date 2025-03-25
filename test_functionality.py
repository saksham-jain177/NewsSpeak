import requests
import json

def test_api():
    BASE_URL = "http://localhost:5000"
    
    print("Testing API Endpoints...")
    
    # Test 1: Scraping
    print("\n1. Testing /api/scrape endpoint")
    response = requests.get(f"{BASE_URL}/api/scrape?company=Tesla")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Scraping successful. Found {len(data['articles'])} articles")
    else:
        print("❌ Scraping failed:", response.json())

    # Test 2: Sentiment Analysis
    print("\n2. Testing /api/sentiment endpoint")
    test_text = "This is a great company with amazing potential"
    response = requests.post(
        f"{BASE_URL}/api/sentiment",
        json={"text": test_text}
    )
    if response.status_code == 200:
        print("✅ Sentiment analysis successful:", response.json())
    else:
        print("❌ Sentiment analysis failed:", response.json())

    # Test 3: Full Report
    print("\n3. Testing /api/report endpoint")
    response = requests.get(f"{BASE_URL}/api/report?company=Tesla")
    if response.status_code == 200:
        data = response.json()
        print("✅ Report generation successful")
        print("Analysis results:", json.dumps(data["Analysis"], indent=2))
    else:
        print("❌ Report generation failed:", response.json())

    # Test 4: TTS Generation
    print("\n4. Testing /api/tts endpoint")
    response = requests.post(
        f"{BASE_URL}/api/tts",
        json={"text": "This is a test message for Hindi conversion"}
    )
    if response.status_code == 200:
        print("✅ TTS generation successful")
        print("Audio file generated")
    else:
        print("❌ TTS generation failed:", response.json())

if __name__ == "__main__":
    test_api()
