import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os

# Import Hugging Face Transformers
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForTokenClassification
)

# Create FastAPI app
app = FastAPI(
    title="Versatile NLP Pipeline",
    description="A comprehensive NLP analysis API for multiple text processing tasks"
)

# Global model caches to improve performance
class ModelCache:
    def __init__(self):
        self.sentiment_model = None
        self.ner_model = None
        self.topic_model = None
        self.summarization_model = None
        self.keyword_model = None

    def get_sentiment_model(self):
        if not self.sentiment_model:
            self.sentiment_model = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        return self.sentiment_model

    def get_ner_model(self):
        if not self.ner_model:
            self.ner_model = pipeline(
                "ner", 
                model="dslim/bert-base-NER"
            )
        return self.ner_model

    def get_topic_model(self):
        if not self.topic_model:
            # Note: You might want to replace with a more specific topic classification model
            self.topic_model = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli"
            )
        return self.topic_model

    def get_summarization_model(self):
        if not self.summarization_model:
            self.summarization_model = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn"
            )
        return self.summarization_model

# Initialize model cache
model_cache = ModelCache()

# Request models
class SentimentRequest(BaseModel):
    text: str

class NERRequest(BaseModel):
    text: str

class TopicClassificationRequest(BaseModel):
    text: str
    candidate_labels: List[str]
    multi_label: Optional[bool] = False

class SummarizationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 150
    min_length: Optional[int] = 50

class KeywordExtractionRequest(BaseModel):
    text: str
    num_keywords: Optional[int] = 5

# Sentiment Analysis Endpoint
@app.post("/sentiment-analysis/")
def analyze_sentiment(request: SentimentRequest):
    """
    Perform sentiment analysis on input text.
    Returns sentiment label and confidence score.
    """
    try:
        sentiment_model = model_cache.get_sentiment_model()
        result = sentiment_model(request.text)[0]
        return {
            "sentiment": result['label'],
            "confidence": result['score']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Named Entity Recognition Endpoint
@app.post("/named-entity-recognition/")
def perform_ner(request: NERRequest):
    """
    Perform Named Entity Recognition on input text.
    Returns list of detected entities with their types.
    """
    try:
        ner_model = model_cache.get_ner_model()
        entities = ner_model(request.text)
        
        # Group consecutive entities
        grouped_entities = []
        current_entity = None
        for entity in entities:
            if entity['entity'].startswith('B-'):
                # Begin of a new entity
                if current_entity:
                    grouped_entities.append(current_entity)
                current_entity = {
                    'word': entity['word'],
                    'entity_type': entity['entity'][2:],
                    'confidence': entity['score']
                }
            elif entity['entity'].startswith('I-'):
                # Inside an entity
                if current_entity and current_entity['entity_type'] == entity['entity'][2:]:
                    current_entity['word'] += f" {entity['word']}"
        
        # Add last entity
        if current_entity:
            grouped_entities.append(current_entity)
        
        return grouped_entities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Topic Classification Endpoint
@app.post("/topic-classification/")
def classify_topic(request: TopicClassificationRequest):
    """
    Perform zero-shot topic classification.
    Allows classification into custom categories.
    """
    try:
        topic_model = model_cache.get_topic_model()
        result = topic_model(
            request.text, 
            request.candidate_labels, 
            multi_label=request.multi_label
        )
        return {
            "text": request.text,
            "classifications": [
                {
                    "label": label, 
                    "score": score
                } for label, score in zip(result['labels'], result['scores'])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Summarization Endpoint
@app.post("/summarization/")
def summarize_text(request: SummarizationRequest):
    """
    Generate a summary of the input text.
    Allows customization of summary length.
    """
    try:
        summarization_model = model_cache.get_summarization_model()
        summary = summarization_model(
            request.text, 
            max_length=request.max_length, 
            min_length=request.min_length,
            do_sample=False
        )[0]['summary_text']
        return {
            "original_text": request.text,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Keyword Extraction Endpoint (using a simple extraction method)
@app.post("/keyword-extraction/")
def extract_keywords(request: KeywordExtractionRequest):
    """
    Extract top keywords from the input text.
    Uses a simple TF-IDF based approach.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        # Create TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 2)
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform([request.text])
        
        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get TF-IDF scores
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Create word-score dictionary
        word_scores = dict(zip(feature_names, tfidf_scores))
        
        # Sort and return top keywords
        sorted_keywords = sorted(
            word_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:request.num_keywords]
        
        return {
            "keywords": [
                {
                    "keyword": keyword, 
                    "score": float(score)
                } for keyword, score in sorted_keywords
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point for running the server
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)