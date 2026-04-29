CODE:

pip install nltk pytube youtube-transcript-api colorama openai

import re
import nltk
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from heapq import nlargest
from urllib.parse import urlparse, parse_qs
from openai import OpenAI

nltk.download('punkt')
nltk.download('stopwords')

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_API_KEY"
)

def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)

    if parsed_url.netloc == 'youtu.be':
        return parsed_url.path[1:]

    if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]

    raise ValueError("Invalid YouTube URL")

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_text_nltk(text, num_sentences=5):
    if not text or text.startswith("Error"):
        return "No valid transcript available."

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())

    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    freq_dist = FreqDist(filtered_words)

    sentences = sent_tokenize(text)

    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq_dist:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq_dist[word]

    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    return " ".join(summary_sentences)

def summarize_with_openai(text):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Summarize this:\n{text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI Error: {str(e)}"

def main():
    youtube_url = input("Enter YouTube URL: ")

    try:
        video_id = extract_video_id(youtube_url)
        transcript = get_transcript(video_id)

        print("\n--- TRANSCRIPT (short preview) ---\n")
        print(transcript[:500], "...")

        print("\n--- NLTK SUMMARY ---\n")
        nltk_summary = summarize_text_nltk(transcript)
        print(nltk_summary)

        print("\n--- OPENAI SUMMARY ---\n")
        ai_summary = summarize_with_openai(transcript)
        print(ai_summary)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
