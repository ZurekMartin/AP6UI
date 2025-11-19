#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import requests
import matplotlib.pyplot as plt
from collections import Counter
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def extract_reviews(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Chyba při načítání URL {url}: {e}")
        return []
    soup = BeautifulSoup(response.text, 'html.parser')
    reviews = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 100]
    if not reviews:
        reviews = [tag.get_text().strip() for tag in soup.find_all(lambda tag: tag.name in ['div', 'section'] and tag.get('class') and any("review" in c.lower() for c in tag.get('class')) and len(tag.get_text().strip()) > 50)]
    return reviews

def generate_wordcloud(data, color_map='viridis'):
    return WordCloud(width=800, height=400, background_color='white', max_words=100, colormap=color_map, prefer_horizontal=0.9, min_font_size=10).generate_from_frequencies(data)

def _create_text_table(ax, title, text):
    ax.axis('off')
    ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
    table = ax.table(cellText=[[text]], cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

def display_results(results, report_id):
    if not results:
        return
    fig, axes = plt.subplots(7, 1, figsize=(12, 20), gridspec_kw={'height_ratios': [0.7, 0.7, 1, 1, 1, 1, 1]})
    plt.rcParams['font.family'] = 'monospace'
    sentiment_summary = f"Positive: {results['sentiment_counts']['positive']}, Neutral: {results['sentiment_counts']['neutral']}, Negative: {results['sentiment_counts']['negative']}"
    url_summary = f"URL: {results['url']}\n\nPočet recenzí: {results['reviews_count']}\n\nSentiment recenzí: {sentiment_summary}"
    _create_text_table(axes[0], "", url_summary)
    axes[1].axis('off')
    axes[1].set_title("Wordcloud sentimentu recenzí", pad=20, fontsize=12, fontweight='bold')
    sentiment_wc = generate_wordcloud(results['sentiment_counts'])
    axes[1].imshow(sentiment_wc, interpolation='bilinear')
    common_text = "\n".join(" , ".join(f"{w} → {c}" for w, c in results['most_common_words'][i:i+5]) for i in range(0, len(results['most_common_words']), 5))
    common_summary = "30 nejpoužívanějších slov z recenzí\n\n" + common_text
    _create_text_table(axes[2], "", common_summary)
    axes[3].axis('off')
    axes[3].set_title("Wordcloud nejčastějších slov z recenzí", pad=20, fontsize=12, fontweight='bold')
    common_wc = generate_wordcloud(dict(results['most_common_words']))
    axes[3].imshow(common_wc, interpolation='bilinear')
    longest_text = "\n".join(" , ".join(f"{w} → {l}" for w, l in results['longest_words'][i:i+5]) for i in range(0, len(results['longest_words']), 5))
    longest_summary = "30 nejdelších slov z recenzí\n\n" + longest_text
    _create_text_table(axes[4], "", longest_summary)
    axes[5].axis('off')
    axes[5].set_title("Wordcloud nejdelších slov z recenzí", pad=20, fontsize=12, fontweight='bold')
    longest_wc = generate_wordcloud(dict(results['longest_words']), color_map='plasma')
    axes[5].imshow(longest_wc, interpolation='bilinear')
    axes[6].axis('off')
    plt.tight_layout()
    filename = f"nlp_analysis_result_{results['url'].split('/')[-2]}_{report_id}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Výsledky analýzy byly uloženy do souboru '{filename}'")

class ReviewAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def analyze_sentiment(self, text):
        return self.sia.polarity_scores(text)

    def clean_text(self, text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return [word for word in word_tokenize(text.lower()) if word not in self.stop_words and word.isalpha()]

    def common_words(self, text, n=30):
        return Counter(self.clean_text(text)).most_common(n)

    def longest_words(self, text, n=30):
        words = list(set(self.clean_text(text)))
        words.sort(key=len, reverse=True)
        return [(word, len(word)) for word in words[:n]]

    def analyze_reviews(self, url):
        reviews = extract_reviews(url)
        if not reviews:
            print(f"Nebyly nalezeny žádné recenze na URL: {url}")
            return None
        all_text = " ".join(reviews)
        sentiments = [self.analyze_sentiment(r) for r in reviews]
        sentiment_counts = {
            'positive': sum(1 for s in sentiments if s['compound'] > 0.05),
            'neutral': sum(1 for s in sentiments if -0.05 <= s['compound'] <= 0.05),
            'negative': sum(1 for s in sentiments if s['compound'] < -0.05)
        }
        return {
            'url': url,
            'reviews_count': len(reviews),
            'sentiment_counts': sentiment_counts,
            'reviews': reviews,
            'sentiments': sentiments,
            'most_common_words': self.common_words(all_text),
            'longest_words': self.longest_words(all_text)
        }

    def analyze_and_display(self, url, report_id):
        results = self.analyze_reviews(url)
        if results:
            display_results(results, report_id)
            return True
        return False

def main():
    analyzer = ReviewAnalyzer()
    urls = [
        "https://www.lttstore.com/products/screwdriver",
        "https://www.ifixit.com/en-eu/products/manta-driver-kit-112-bit-driver-kit#reviews",
        "https://topofthemornincoffee.com/en-eu/products/chocolate-irish-cream-flavored-coffee"
    ]
    for idx, url in enumerate(urls, 1):
        print(f"\nAnalyzuji URL: {url}")
        analyzer.analyze_and_display(url, idx)

if __name__ == "__main__":
    main()
