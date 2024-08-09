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

        # ad_text = "Ad will be here"
        yahoo_search_url = f'https://search.yahoo.com/search?p={ad_text}&weight={sentiment}'

        return f'<div class="ios_fix" style="margin: 0;padding: 0;width: 100%;min-width: 100%;"><div class="container" id="container" style="margin: 0;padding: 0;position: relative; overflow: hidden; font-family: \'\"Arial\", sans-serif\'\'; height: 115px;background: #fff;"><ul class="list_1" style="margin: 0;padding: 0;list-style: none;width: 100%;padding-top: 16px;"><li style="margin: 0;padding: 0;position: relative;overflow: hidden;*float: left;width: 100%;margin-top: 0;"><div class="content" style="margin: 0;padding: 0 30px 0 34px;position: relative;display: block;overflow: hidden;transition: all 300ms ease;border-bottom-width: 2px;background: #fff;border-bottom-style: solid;border-bottom-color: #80b280;"><div class="num" style="margin: 0;padding: 0;width: 24px;height: 24px;position: absolute;left: 5px;top: 50%;margin-top: -12px;transition: all 300ms ease;text-align: center;line-height: 23px;font-family: arial, Arial, sans-serif;font-size: 14px;color: #000;font-weight: Bold;">1.</div><div class="arrow" style="margin: 0;padding: 0;top: 50%;margin-top: -13px;position: absolute;overflow: hidden;height: 26px;line-height: 26px;right: 10px;font-weight: bold;transition: all 300ms ease;font-size: 20px;color: #80b280;">❯</div><table width="100%" cellpadding="0" cellspacing="0" border="0" align="left" style="margin: 0;padding: 0;table-layout: fixed;width: 100%;word-wrap: break-word;"><tbody style="margin: 0;padding: 0;"><tr style="margin: 0;padding: 0;"><td width="100%" align="left" valign="middle" style="border:none; margin: 0;padding: 0;height: 80px;"><div class="anchortext" style="margin: 0;padding: 0;"><a href={yahoo_search_url} id="dk1" name="dk1" target="_blank" onclick="" style="margin: 0;padding: 0;outline: 0;text-decoration: none;cursor: pointer;line-height: 22px;overflow: hidden;word-wrap: break-word;display: block;transition: all 300ms ease;max-height: 44px;font-size: 17px;color: #000;font-family: arial, Arial, sans-serif;text-transform: none;font-weight: &nbsp;">{ad_text}</a></div></td></tr></tbody></table>  </div><a href="https://related.lifestyle-insights.com/search.cfm?ule=2724&amp;prvep=g4ztEd5Gptti639Yk9qpGA%3D%3D&amp;ktr=1&amp;vi=1723193203852638298&amp;cq=GbdrWg9dQvNd0f1a5Xqd8WQf95qaf&amp;vsid=3579799101500997&amp;sc=KA&amp;oksu=224&amp;prid=8PRVCXX19&amp;cid=8CU48T46H&amp;pid=8PO7O9Y4F&amp;crid=738620718&amp;https=1&amp;rms=1723197183&amp;size=336x280&amp;ksu=224&amp;ugd=4&amp;tsid=7&amp;asn=45820&amp;eptitle=3mxLnpJGE1yJ&amp;radepth=0&amp;kwep=Q4U5PZHT%26U%2FJPB3aa%26UWPFHzZGtGGE%26UgPz%26U5PZHT&amp;oscar=1&amp;tsce=L586-S586&amp;ssld=%7B%22QQNN%22%3A%22ZR%22%2C%22QQN75%22%3A%22N1LO8kk%22%2C%22QQ8E%22%3A%22f19i%3AG1NA%3AAhhH%3A9999%3A9999%3A9999%3A9999%3A9999%22%2C%22QQQN%22%3A%22%3DDs%22%7D&amp;vgd_ifrmode=00&amp;&amp;fp=0-5pYlxpKR9Z9EEVTslDYmMjpMYrbtcIF_Imog9FA_8O94NIVTuqk9XGM0qOi89PlwaNn-mZHbznC3v5ER_26tz5uCogRVOZreeLhWu5o80QHU-0g2z_wiwh_guGeXQ3Jgi1u8I9Y_8%3D&amp;cme=GiuUGRkRGxz4-hWAEcvJRqEsyPJgZ7QvlMv-dEVNON-UyWMXememrwSBwLxMtZIwFqqEUC-71JcdHOIUzj94QEX2SzBopGfYiI0P5wr_VExxX6pxcz7IpDITaoMVKtdvekZ6XIFc9wG3avR6agQ_kxAnUfRNsZcizFTCxCd7sh-c7Fr4FvXs2ggQ3JrximzZJtCaW4RHgZIOqoimqtm0tNgNQUr3QzwIpUWyzeeyKlw%3D%7C%7CO5mjPGhVLdMRUkH1eGi2Gg%3D%3D%7CuwFQFUQk4gWAa-UQbotTIITQMAoZPqpSBxV7DOpt8Yf42hRVRYlGM2kbDfyJC46ZWS71MhEnScBLI3z5cOSpBg%3D%3D%7CcPcb3VhU0BVjXgWFWEAzinttU1oq1ouO%7Cu8A6SM53vAcxcExhY54hOIM822uWnNPY%7Ck1j4PdXd7Pzzgr_Iw3h8eUC-qQArclPQ%7C&amp;bdrid=460&amp;bd=5.5%231329%232056&amp;lgpl=eQ7L8O%3ArJk%2F8O%7CGO171%3AQOfvzxjj~8xLjMjvF9~myJLEYv9.HX~eBMJ-Nv9.i~e8QMQOvfuW~ONfvf~QNOvNLk~L1Jvf%2C9%2Cou~eM1QzvuAAAX~ejfLMQOvf9fH9W9i9f~8xLjMGvXXu.AF~xLjM7UNvu~7M7QvAAF-fW9~Q7OvH9HiFA9Wi9~c0v.*Q1jJ.*~j1Q7v%24%7Bj1Q7Mkj1y%7D~e8QMxLjMGv9.hf~8EvuwUhqOBzHpe%2FNNPD9%3DTwFB~kGGv9~e8QMxLjMjvH9~L88Ex1vAFAW%2CAFHf~J7vuA~LNvu~LJkMLvI~LEQMQOvf9fH9W9Wf9~e8QMGvWXf.u~xLjMGvu.Wu~ejfLMxLjMGv9.9u~ejfLMxLjMe8vFW~xLjM7e8vAX~xLjMjvF9~yN17vX9uuHf~GGvuiF~eev9~jfLMGvu999~JLEYv9.HX~ejfLMxLjMUNvu49~GYvu~LUJvH%2C9%2Cou~L1OEv9.999%2C9~1AEMGvF.FW%2ChW.XF~Q8OvOF9kGJXG1X9ffAiOHOkAF9HXAHWXGhiF~QOv9~x8OvffOXse87Z26WQiwe_f~G7OvufAuAuAuAH99XXi9WFhWfhAi9iAuWhui9H9iAHifFAWHuHiFiHAHXHWFXuHFXhHhiX9HWWufAHAWAWfhA9AFiu9X9fX9ifFA9Fi9uHFHhHhhfXAAfHW~eBxv9.i~OfEMjvu9~AENkvu999~x8YvAXiWfu~LM7QvA99-fX9%2CAf9-u99%2CAAF-fW9~myMYQwv9.HX~1EEMzvzmzM1EE~OYYMQ7LyvE8zz1NjJ~eLMxLjMGvhfA~OfEMGvu~myOfEMGv9.iX~exLjMGvu.i~GxyOvA~QQvHuH-WiF~NNvZR~x8Bvu99~NJvu~LEQMGvhW.XF~exLjMjvX9~%3DVvAf9A~LUBEv9.999%2C9~UGMxNvof~z7QvA~c0fv.*Q1jJ.*~N7vN1LO8kk~J-EQNmLJvou~G1Q8QfvuiF~GO7vuhfAuifWu9~G1Q8QuvuiF~8QDJkv%24%7BLJkLJQwMNmxz7JL%7D~8exLjMGv9.XA~0sv9~8Q8kv9~G8Ov9.HX~ONvX~ejfLMGvF.FW~8exLjMjvX9~%24%7B%3Dj8Jz73Tmy%7D~8GNvu~zQlvA~7yQvA99-fX9%23%40Af9-u99%23%40AAF-fW9~GQ1v%2Fu~GQGvA~GQEvf~7Y-vfHi~Y-GU7v9~Y-wYQvfF%7CQNQeJL%3AWFA%7Cw8Yyjy%3AC909oa9C9ob9%7CNkxO%3AfA9Aff%7CmE7mx7%3A9%7C5OQEL%3Au%7CjfQwjO%3Au%7C1UN8E%3AuH.ii.uFh.9%7CmNw%3A9%7CLJzQ8lJ%3AiAhMu9Wi%7CQNLMw%3AuAfi%7CQNLMB%3Af9XF%7CJN7%3AHy&amp;kct=333&amp;ure=1" class="dak1 anchorhref" target="_blank" style="margin: 0;padding: 0;outline: 0;text-decoration: none;cursor: pointer;position: absolute;display: block;width: 100%;height: 100%;left: 0;top: 0;overflow: hidden;z-index: 999;background: url(data:image/png;base64,ivborw0kggoaaaansuheugaaaaeaaaabcayaaafoevqfaaaagxrfwhrtb2z0d2fyzqbbzg9izsbjbwfnzvjlywr5ccllpaaaaa1jrefuenpj+p//pwmacpwc/njesraaaaaasuvork5cyii=) right top no-repeat: ;"></a></li></ul><div class="header_title" style="margin: 0;padding: 0;position: absolute;top: 0px;left: 0px;font-family: arial, Arial, sans-serif;text-transform: none;font-weight: normal;color: #7a7272;font-size: 14px;line-height: 16px;">Search for</div><div class="header_wrap" style="margin: 0;padding: 0;position: absolute;bottom: 1px;left: 1px;font-family: Arial,sans-serif;z-index: 999;display: flex;align-items: center;flex-wrap: wrap;height: 14px;overflow: hidden;">    <div class="header" style="margin: 0;padding: 0;height: 12px;width: 14px;background-color: #142c58;color: #fff;font-size: 7px;border-radius: 2px;text-align: center;font-weight: bold;display: flex;align-items: center;justify-content: center;">Ad</div>    <div class="sep" style="margin: 0 3px;padding: 0;background: #7a7272;width: 1px;height: 12px;"></div>    <div class="brand_txt" id="brand_txt" style="margin: 0;padding: 0;font-size: 11px;font-family: arial, Arial, sans-serif;"><a href="https://lifestyle-insights.com/" target="_blank" style="margin: 0;padding: 0;outline: 0;text-decoration: none;cursor: pointer;color: #7a7272;">Lifestyle Insights</a></div></div></div></div>'

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

@app.route('/analyze_content', methods=['POST'])
def analyze_content():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "URL is required"}), 400

    paragraph_indices, segment_info, segment_weights = analyze_and_cache(url)
    if paragraph_indices is None:
        return jsonify({"error": segment_info}), 500

    response = {
        "paragraph_indices": paragraph_indices,
        "segment_info": segment_info,
        "segment_weights": segment_weights.tolist()
    }
    return jsonify(response)

@app.route('/generate_ads_only', methods=['POST'])
def generate_ads_only():
    data = request.json
    segment_info = data.get('segment_info')
    if not segment_info:
        return jsonify({"error": "Segment information is required"}), 400

    ad_generator = AdGeneratorAI()
    ads = [ad_generator.generate_ad(content, sentiment) for (content, sentiment) in segment_info]

    response = {
        "ads": ads
    }
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
