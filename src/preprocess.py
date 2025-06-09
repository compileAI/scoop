from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
from typing import List, Iterable

from google import genai
import pandas as pd
import spacy
from supabase import create_client
from tqdm.auto import tqdm

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("preproc")

# spaCy pipeline – blank model + sentencizer + lemmatizer
def build_nlp() -> spacy.language.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
    log.info("spaCy pipeline ready: %s", nlp.pipe_names)
    return nlp

nlp = build_nlp()

# tokenisation helpers
def _clean_token(tok: spacy.tokens.Token) -> str | None:
    """Return a lowercase lemma or None if we should skip the token."""
    if tok.is_alpha and not tok.is_punct and tok.text.lower() not in nlp.Defaults.stop_words:
        return tok.lemma_.lower()
    return None

def tokenize_texts(texts: Iterable[str], batch_size: int = 512) -> List[List[str]]:
    """
    Lemmatise + stop-word-filter an iterable of texts.
    Returns a list-of-lists in the input order.
    """
    out: List[List[str]] = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        out.append([t for t in (_clean_token(tok) for tok in doc) if t])
    return out

def read_from_supabase(table_name: str) -> pd.DataFrame:
    """Read a table from Supabase and map columns to expected format."""
    df = supabase.table(table_name).select("*").execute().data
    df = pd.DataFrame(df)
    
    # Map columns to expected format
    column_mapping = {
        'id': 'id',  # already matches
        'published': 'date',
        'title': 'title',  # already matches
        'content': 'text'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required = {"id", "date", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing!r}")
    
    return df

# main dataframe pre-processor
def preprocess_articles(df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    """
    Expect columns: id | date | title | text
    Adds: sentences, sentence_tokens, sentence_embds, sentence_counts
    """
    log.info("Dataframe shape before processing: %s", df.shape)
    required = {"id", "date", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"article_df missing columns: {missing!r}")

    # sentence segmentation (title + body)
    titles = df["title"].tolist()
    bodies = df["text"].tolist()

    # split bodies into sentences once, batched
    body_sents: List[List[str]] = []
    with tqdm(total=len(bodies), desc="Sentence split") as bar:
        for doc in nlp.pipe(bodies, batch_size=batch_size):
            body_sents.append([s.text.strip() for s in doc.sents if len(s) > 1])
            bar.update(1)

    # title sentences are just the title itself
    title_sents = [[t] for t in titles]

    sentences = [ts + bs for ts, bs in zip(title_sents, body_sents)]
    sentence_counts = [len(s) for s in sentences]

    # tokenisation (same order)
    flat_texts = [s for doc in sentences for s in doc]
    flat_tokens = tokenize_texts(flat_texts, batch_size=batch_size)
    sentence_tokens: List[List[List[str]]] = []
    idx = 0
    for count in sentence_counts:
        sentence_tokens.append(flat_tokens[idx : idx + count])
        idx += count

    embeddings = []
    for batch in tqdm(
        sentences, desc="Embedding sentences", total=len(sentences)
    ):
        batch_embeddings = client.models.embed_content(
            model="text-embedding-004",
            contents=batch,
        )
        embeddings.append(batch_embeddings)

    # assemble dataframe
    out = df.copy()
    out["sentences"] = sentences
    out["sentence_counts"] = sentence_counts
    out["sentence_tokens"] = sentence_tokens
    out["sentence_embds"] = embeddings

    log.info("Completed preprocessing - final shape: %s", out.shape)
    return out

def read_todays_articles() -> pd.DataFrame:
    """Read articles created today from Supabase."""
    today = datetime.now(timezone.utc).date()
    df = supabase.table("source_articles") \
        .select("*") \
        .gte('created_at', f"{today}T00:00:00Z") \
        .lt('created_at', f"{today}T23:59:59Z") \
        .execute().data
    
    df = pd.DataFrame(df)
    
    # Map columns to expected format
    column_mapping = {
        'id': 'id',
        'published': 'date',
        'title': 'title',
        'content': 'text'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required = {"id", "date", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing!r}")
    
    return df

def read_all_articles() -> pd.DataFrame:
    """Read all articles from Supabase for cold-start processing."""
    df = supabase.table("source_articles") \
        .select("*") \
        .execute().data
    
    df = pd.DataFrame(df)
    
    # Map columns to expected format
    column_mapping = {
        'id': 'id',
        'published': 'date',
        'title': 'title',
        'content': 'text'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required = {"id", "date", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing!r}")
    
    return df

# quick CLI demo
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Process articles from Supabase')
    parser.add_argument('--cold_start', action='store_true', help='Process all articles (cold start)')
    args = parser.parse_args()
    
    if args.cold_start:
        df = read_all_articles()
    else:
        df = read_todays_articles()
    
    proc_df = preprocess_articles(df)
