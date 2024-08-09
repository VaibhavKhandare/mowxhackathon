import requests
from bs4 import BeautifulSoup
import spacy
import numpy as np
from spacytextblob.spacytextblob import SpacyTextBlob
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import openai
from flask import Flask, request, jsonify, send_file
from cachetools import cached, TTLCache
from flask_cors import CORS  # Import CORS
import matplotlib.pyplot as plt
import io
from dotenv import load_dotenv
import os
from collections import Counter
import string

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI')

# Cache setup: Using TTLCache with maxsize of 100 entries and TTL of 1 day
cache = TTLCache(maxsize=100, ttl=86400)

# Initialize Flask app

app = Flask(__name__)
CORS(app)

class DataCollectionAI:
    def __init__(self, url):
        self.url = url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

    def fetch_content(self):
        response = requests.get(self.url, headers=self.headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            return soup, paragraphs
        else:
            return None, f"Failed to retrieve content. Status code: {response.status_code}"

class NLPAI:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe('spacytextblob')
        self.page_vector = None
        self.overall_sentiment = 0.0

    def compute_page_vector_and_sentiment(self, content):
        doc = self.nlp(content)
        vectors = [token.vector for token in doc if not token.is_stop and token.is_alpha]
        self.page_vector = np.mean(vectors, axis=0) if vectors else None
        self.overall_sentiment = doc._.blob.sentiment.polarity
        return self.page_vector, self.overall_sentiment

    def analyze_content(self, content):
        self.compute_page_vector_and_sentiment(content)

        doc = self.nlp(content)
        segments = []
        weights = []

        for sent in doc.sents:
            sentiment = sent._.blob.sentiment.polarity
            sentiment_diff = 1 - abs(sentiment - self.overall_sentiment) / 2
            length_score = len(sent.text.split()) / len(doc.text.split())
            relevance_score = self.compute_similarity(sent.text)
            weight = (sentiment_diff * 0.2 + length_score * 0.29 + relevance_score * 0.5) * (1 - 0 * sent.start_char / len(doc.text))
            segments.append((sent.text, sentiment, weight))
            weights.append(weight)
        
        smoothed_weights = gaussian_filter1d(weights, sigma=2)

        return segments, smoothed_weights

    def compute_similarity(self, segment_text):
        if self.page_vector is not None:
            segment_doc = self.nlp(segment_text)
            segment_vectors = [token.vector for token in segment_doc if not token.is_stop and token.is_alpha]
            if segment_vectors:
                segment_vector = np.mean(segment_vectors, axis=0)
                similarity = np.dot(self.page_vector, segment_vector) / (np.linalg.norm(self.page_vector) * np.linalg.norm(segment_vector))
                return similarity
        return 0.0

class AdGeneratorAI:
    def __init__(self):
        self.prompt_template = (
            "You are an expert copywriter tasked with creating short and crisp search ads. Based on the following content segment and its sentiment, craft a short, engaging search ad within 6-7 words\n\n"
            "Content: {content}\n"
            "Sentiment Score: {sentiment}\n\n"
            "Search Ad:"
        )
    def generate_ad(self, content, sentiment):
        prompt = self.prompt_template.format(content=content, sentiment=sentiment)
        response = openai.ChatCompletion.create(
            model="gpt-4", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant who generates ads"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        ad_text = response.choices[0].message['content'].strip()

        yahoo_search_url = f'https://search.yahoo.com/search?p={ad_text}&weight={sentiment}'

        return f'<a href="{yahoo_search_url}" target="_blank"><div>{ad_text}</div></a>'

class AdPlacementStrategyAI:
    def __init__(self, segments, smoothed_weights):
        self.segments = segments
        self.smoothed_weights = smoothed_weights

    def suggest_ad_placements(self, num_ads=None):
        # Identify local maxima in the smoothed weights
        peaks, _ = find_peaks(self.smoothed_weights, distance=3, prominence=0.01)
        
        # Sort peaks by their weight and select top ones
        top_peaks = sorted(peaks, key=lambda x: self.smoothed_weights[x], reverse=True)


        if num_ads is None:
            # Determine the number of ads based on the page size
            num_ads = max(1, int(len(self.smoothed_weights) / 30))

        if len(top_peaks) < num_ads:
            # If there are fewer peaks than required ads, place ads at equally spaced paragraphs
            total_segments = len(self.smoothed_weights)
            top_peaks = np.linspace(0, total_segments - 1, num=num_ads, dtype=int)
        else:
            top_peaks = top_peaks[:num_ads]

        return top_peaks

@cached(cache)
def analyze_and_cache(url):
    dc_ai = DataCollectionAI(url)
    soup, paragraphs = dc_ai.fetch_content()
    if soup is None:
        return None, paragraphs

    nlp_ai = NLPAI()
    content = "\n".join([p.get_text() for p in paragraphs])
    segments, smoothed_weights = nlp_ai.analyze_content(content)

    aps_ai = AdPlacementStrategyAI(segments, smoothed_weights)
    top_peaks = aps_ai.suggest_ad_placements()


    paragraph_indices = []
    segment_info = []
    segment_weights = smoothed_weights

    # Iterate over each paragraph and its index
    for i, para in enumerate(paragraphs):
        para_text = para.get_text().strip()
        para_doc = nlp_ai.nlp(para_text)  # Process paragraph text with NLP

        # Split paragraph into sentences
        para_sentences = [sent.text.strip() for sent in para_doc.sents]

        # Iterate over sentences and their indices within the paragraph
        for j, sent in enumerate(para_sentences):
            # Calculate global sentence index (if segments are global)
            global_index = sum(len(paragraphs[k].get_text().strip().split('.')) for k in range(i)) + j
            
            # Check if the global sentence index is in the top_peaks list
            if global_index in top_peaks:
                paragraph_indices.append(i)
                segment_info.append((segments[global_index][0], segments[global_index][2]))

    cache[url] = {
        'paragraph_indices': paragraph_indices,
        'segment_info': segment_info,
        'segment_weights' : smoothed_weights
    }

    return paragraph_indices, segment_info, smoothed_weights

@app.route('/generate_ads', methods=['POST'])
def generate_ads():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "URL is required"}), 400

    cache_data = cache.get(url)
    if cache_data:
        paragraph_indices = cache_data['paragraph_indices']
        segment_info = cache_data['segment_info']
        segment_weights = cache_data["segment_weights"]
    else:
        paragraph_indices, segment_info, segment_weights = analyze_and_cache(url)
        if paragraph_indices is None:
            return jsonify({"error": segment_info}), 500

    ad_generator = AdGeneratorAI()
    ads = [ad_generator.generate_ad(content, sentiment) for (content, sentiment) in segment_info]

    response = {
        "paragraph_indices": paragraph_indices,
        "ads": ads,
        "segment_weights": segment_weights.tolist()
    }
    return jsonify(response)

@app.route('/get_weights', methods=['POST'])
def get_weights():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "URL is required"}), 400

    cache_data = cache.get(url)
    if cache_data:
        paragraph_indices = cache_data['paragraph_indices']
        segment_info = cache_data['segment_info']
        segment_weights = cache_data["segment_weights"]
    else:
        paragraph_indices, segment_info, segment_weights = analyze_and_cache(url)
        if paragraph_indices is None:
            return jsonify({"error": segment_info}), 500

    response = {
        "paragraph_indices": paragraph_indices,
        "segment_weights": segment_weights
    }
    print(len(segment_weights))
    return jsonify(response)


@app.route("/")
def index():
    return "Hello, World!"



@app.route('/clear_cache')
def clear_cache():
    cache.clear()
    return jsonify({"status": "Cache cleared!"})


if __name__ == "__main__":
    app.run(debug=True) 
