from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import time
from typing import List, Iterable

from google import genai
from google.genai import types
import pandas as pd
import spacy
from supabase import create_client
from tqdm.auto import tqdm
from pinecone import Pinecone

from dotenv import load_dotenv

load_dotenv()

from data_types import (
    SUPABASE_ARTICLES_TABLE,
    SUPABASE_CHUNKS_TABLE,
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("scraped-sources-gemini")

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,  # Back to INFO now that we've debugged
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress verbose HTTP logging from supabase/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

log = logging.getLogger("preproc")

# spaCy pipeline – blank model + sentencizer (removed lemmatizer to avoid lookup table issues)
def build_nlp() -> spacy.language.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    # Note: Removed lemmatizer as it requires lookup tables not available in blank model
    log.info("spaCy pipeline ready: %s", nlp.pipe_names)
    return nlp

nlp = build_nlp()

# tokenisation helpers
def _clean_token(tok: spacy.tokens.Token) -> str | None:
    """Return a lowercase token or None if we should skip the token."""
    if tok.is_alpha and not tok.is_punct and tok.text.lower() not in nlp.Defaults.stop_words:
        return tok.text.lower()  # Use text instead of lemma since we don't have lemmatizer
    return None

def tokenize_texts(texts: Iterable[str], batch_size: int = 512) -> List[List[str]]:
    """
    Tokenize + stop-word-filter an iterable of texts.
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
    
    # Handle empty DataFrame
    if len(df) == 0:
        log.warning("Empty DataFrame provided, returning empty processed DataFrame")
        return pd.DataFrame(columns=['id', 'date', 'title', 'text', 'sentences', 'sentence_counts', 'sentence_tokens', 'sentence_embds'])
    
    required = {"id", "date", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"article_df missing columns: {missing!r}")

    # Clean None values in title and text columns to prevent spaCy errors
    df_clean = df.copy()
    df_clean["title"] = df_clean["title"].fillna("")
    df_clean["text"] = df_clean["text"].fillna("")
    
    # Log how many None values were cleaned
    title_none_count = df["title"].isnull().sum()
    text_none_count = df["text"].isnull().sum()
    if title_none_count > 0 or text_none_count > 0:
        log.info(f"Cleaned {title_none_count} None titles and {text_none_count} None text values")

    # sentence segmentation (title + body)
    titles = df_clean["title"].tolist()
    bodies = df_clean["text"].tolist()

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

    # Embedding with proper batching to respect Google API limits (max 100 requests per batch)
    # API limit: 150 requests per minute, so we need more conservative timing to be safe
    embeddings = []
    max_batch_size = 100  # Google API limit
    requests_per_minute = 120  # More conservative rate limit (leave bigger buffer)
    delay_between_requests = 60.0 / requests_per_minute  # 0.5 seconds between requests
    
    for i, batch in enumerate(tqdm(sentences, desc="Embedding sentences", total=len(sentences))):
        # Add delay after first request to respect rate limits
        if i > 0:
            time.sleep(delay_between_requests)
            
        try:
            if len(batch) <= max_batch_size:
                # Batch fits within API limit
                batch_embeddings = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="CLUSTERING", output_dimensionality=768)
                )
                # Extract the actual embedding values correctly
                actual_embeddings = [emb.values for emb in batch_embeddings.embeddings]
                embeddings.append(actual_embeddings)
            else:
                # Split large batches into smaller chunks
                article_embeddings = []
                for j, chunk_start in enumerate(range(0, len(batch), max_batch_size)):
                    # Add delay between chunk requests too
                    if j > 0:
                        time.sleep(delay_between_requests)
                        
                    chunk = batch[chunk_start:chunk_start + max_batch_size]
                    chunk_embeddings = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=chunk,
                        config=types.EmbedContentConfig(task_type="CLUSTERING", output_dimensionality=768)
                    )
                    actual_embeddings = [emb.values for emb in chunk_embeddings.embeddings]
                    article_embeddings.extend(actual_embeddings)
                embeddings.append(article_embeddings)
        except Exception as e:
            if "429" in str(e) or "RATE_LIMIT_EXCEEDED" in str(e):
                log.warning(f"Rate limit hit, waiting 60 seconds before retrying...")
                time.sleep(60)  # Wait a full minute
                
                # Retry the same batch with more conservative approach
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        if len(batch) <= max_batch_size:
                            batch_embeddings = client.models.embed_content(
                                model="text-embedding-004",
                                contents=batch,
                            )
                            actual_embeddings = [emb.values for emb in batch_embeddings.embeddings]
                            embeddings.append(actual_embeddings)
                            break  # Success, exit retry loop
                        else:
                            # Handle large batch retry
                            article_embeddings = []
                            for chunk_start in range(0, len(batch), max_batch_size):
                                chunk = batch[chunk_start:chunk_start + max_batch_size]
                                
                                # Retry each chunk if needed
                                chunk_retry_count = 0
                                while chunk_retry_count < max_retries:
                                    try:
                                        chunk_embeddings = client.models.embed_content(
                                            model="text-embedding-004",
                                            contents=chunk,
                                        )
                                        actual_embeddings = [emb.values for emb in chunk_embeddings.embeddings]
                                        article_embeddings.extend(actual_embeddings)
                                        break  # Success, exit chunk retry loop
                                    except Exception as chunk_e:
                                        if "429" in str(chunk_e) or "RATE_LIMIT_EXCEEDED" in str(chunk_e):
                                            chunk_retry_count += 1
                                            log.warning(f"Chunk retry {chunk_retry_count}/{max_retries}, waiting 30 seconds...")
                                            time.sleep(30)
                                        else:
                                            raise chunk_e
                                
                                # Add delay between chunks
                                if chunk_start + max_batch_size < len(batch):
                                    time.sleep(delay_between_requests * 2)  # Double delay for safety
                            embeddings.append(article_embeddings)
                            break  # Success, exit retry loop
                    except Exception as retry_e:
                        if "429" in str(retry_e) or "RATE_LIMIT_EXCEEDED" in str(retry_e):
                            retry_count += 1
                            log.warning(f"Retry {retry_count}/{max_retries}, waiting {60 * retry_count} seconds...")
                            time.sleep(60 * retry_count)  # Exponential backoff
                        else:
                            raise retry_e
                
                if retry_count >= max_retries:
                    log.error(f"Failed to process batch after {max_retries} retries")
                    raise Exception(f"Max retries exceeded for batch")
            else:
                raise  # Re-raise if it's not a rate limit error

    # assemble dataframe using the cleaned data
    out = df_clean.copy()
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
    
    # Debug: Print available columns
    log.info(f"Available columns from Supabase: {list(df.columns)}")
    log.info(f"Number of rows returned: {len(df)}")
    if len(df) > 0:
        log.info(f"Sample row keys: {list(df.iloc[0].keys()) if len(df) > 0 else 'No data'}")
    
    # Handle empty DataFrame
    if len(df) == 0:
        log.warning("No articles found for today. Consider using --cold_start to process all articles.")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['id', 'date', 'title', 'text'])
    
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
    # First, count the total number of articles
    count_result = supabase.table("source_articles").select("*", count="exact").execute()
    total_count = count_result.count
    
    log.info(f"Found {total_count} total articles in database")
    
    if total_count == 0:
        log.warning("No articles found in database.")
        return pd.DataFrame(columns=['id', 'date', 'title', 'text'])
    
    # Use pagination to fetch all articles (Supabase has a 1000 row limit)
    all_articles = []
    batch_size = 1000
    
    for offset in range(0, total_count, batch_size):
        end_range = min(offset + batch_size - 1, total_count - 1)
        log.info(f"Fetching articles {offset} to {end_range} of {total_count}")
        
        batch_result = supabase.table("source_articles") \
            .select("*") \
            .range(offset, end_range) \
            .execute()
        
        all_articles.extend(batch_result.data)
    
    df = pd.DataFrame(all_articles)
    
    # Debug: Print available columns
    log.info(f"Available columns from Supabase: {list(df.columns)}")
    log.info(f"Number of rows returned: {len(df)}")
    if len(df) > 0:
        log.info(f"Sample row keys: {list(df.iloc[0].keys()) if len(df) > 0 else 'No data'}")
    
    # Handle empty DataFrame
    if len(df) == 0:
        log.warning("No articles found in database.")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['id', 'date', 'title', 'text'])
    
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

def ensure_pinecone_namespace_exists(namespace: str = "chunks") -> None:
    """Ensure the specified namespace exists in the Pinecone index."""
    try:
        # Check if namespace exists by trying to describe index stats for the namespace
        stats = pinecone_index.describe_index_stats()
        if namespace not in stats.get('namespaces', {}):
            log.info(f"Creating Pinecone namespace '{namespace}'")
            # Create namespace by upserting a dummy vector with non-zero values and then deleting it
            dummy_vector = {
                'id': 'dummy_init_vector',
                'values': [0.1] * 768,  # Non-zero values for gemini-embedding-001 (768-dimensional)
                'metadata': {'init': True}
            }
            pinecone_index.upsert(vectors=[dummy_vector], namespace=namespace)
            # Delete the dummy vector
            pinecone_index.delete(ids=['dummy_init_vector'], namespace=namespace)
            log.info(f"Pinecone namespace '{namespace}' created successfully")
        else:
            log.info(f"Pinecone namespace '{namespace}' already exists")
    except Exception as e:
        log.error(f"Error ensuring Pinecone namespace exists: {e}")
        raise

def write_processed_articles_to_supabase_and_pinecone(proc_df: pd.DataFrame) -> None:
    """
    Write processed articles to Supabase tables and embeddings to Pinecone.
    
    Args:
        proc_df: DataFrame with columns: id, date, title, text, sentences, 
                sentence_counts, sentence_tokens, sentence_embds
    """
    log.info(f"Writing {len(proc_df)} processed articles to Supabase and Pinecone")
    
    # Ensure the Pinecone namespace exists
    ensure_pinecone_namespace_exists("source_article_clustering_chunks")
    
    total_chunks = 0
    
    for idx, row in tqdm(proc_df.iterrows(), total=len(proc_df), desc="Writing articles"):
        try:
            # Prepare data for cleaned_source_articles table
            article_data = {
                'source_article_id': row['id'],  # Reference to the original source_articles ID
                'sentence_tokens': row['sentence_tokens'],  # Will be stored as JSON in the database
                'sentence_counts': int(row['sentence_counts']),
                'title': row['title'],
                'text': row['text'],
                'date': row['date']
            }
            
            # Upsert into cleaned_source_articles (insert or update if exists)
            # Use on_conflict parameter to specify which column to use for conflict resolution
            article_result = supabase.table(SUPABASE_ARTICLES_TABLE).upsert(
                article_data, 
                on_conflict="source_article_id"
            ).execute()
            
            if not article_result.data:
                log.error(f"Failed to upsert article {idx}")
                continue
                
            source_article_id = row['id']  # Original string ID, now the unique identifier
            log.debug(f"Upserted article with source_article_id: {source_article_id}")
            
            # Process each sentence as a chunk
            sentences = row['sentences']
            embeddings = row['sentence_embds']
            
            # First, delete any existing chunks for this article (in case we're reprocessing)
            existing_chunks = supabase.table(SUPABASE_CHUNKS_TABLE).select("chunk_id").eq('source_article_id', source_article_id).execute()
            if existing_chunks.data:
                # Delete old Pinecone vectors
                old_chunk_ids = [str(chunk['chunk_id']) for chunk in existing_chunks.data]
                pinecone_index.delete(ids=old_chunk_ids, namespace="source_article_clustering_chunks")
                log.debug(f"Deleted {len(old_chunk_ids)} old Pinecone vectors for source_article_id: {source_article_id}")
            
            # Delete existing chunks from database
            supabase.table(SUPABASE_CHUNKS_TABLE).delete().eq('source_article_id', source_article_id).execute()
            
            # Prepare batch data for chunks
            chunk_data = []
            pinecone_vectors = []
            
            for sentence_idx, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
                chunk_record = {
                    'text': sentence,
                    'source_article_id': source_article_id  # Reference to original source_articles ID
                }
                chunk_data.append(chunk_record)
            
            # Batch insert chunks to get chunk_ids
            if chunk_data:
                chunk_result = supabase.table(SUPABASE_CHUNKS_TABLE).insert(chunk_data).execute()
                
                if chunk_result.data:
                    # Now prepare Pinecone vectors with the actual chunk_ids
                    for chunk_record, embedding in zip(chunk_result.data, embeddings):
                        chunk_id = chunk_record['chunk_id']
                        
                        # embedding should already be a list of float values from the preprocessing
                        embedding_values = list(embedding)
                            
                        pinecone_vectors.append({
                            'id': str(chunk_id),
                            'values': embedding_values,
                            'metadata': {
                                'source_article_id': source_article_id,    # Reference to source_articles.id
                                'text': chunk_record['text'][:1000]        # Limit metadata text length
                            }
                        })
                    
                    # Batch upsert to Pinecone
                    if pinecone_vectors:
                        pinecone_index.upsert(
                            vectors=pinecone_vectors,
                            namespace="source_article_clustering_chunks"
                        )
                        total_chunks += len(pinecone_vectors)
                        log.debug(f"Upserted {len(pinecone_vectors)} vectors to Pinecone for source_article_id: {source_article_id}")
                else:
                    log.error(f"Failed to insert chunks for source_article_id: {source_article_id}")
            
        except Exception as e:
            log.error(f"Error processing article {idx}: {e}")
            continue
    
    log.info(f"Successfully wrote {len(proc_df)} articles and {total_chunks} chunks to database and Pinecone")

def main_daily():
    """Process articles created today."""
    df = read_todays_articles()
    proc_df = preprocess_articles(df)
    write_processed_articles_to_supabase_and_pinecone(proc_df)

def read_articles_from_days(days: int) -> pd.DataFrame:
    """Read articles from the last N days from Supabase."""
    from datetime import datetime, timedelta
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Format dates for Supabase query
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    log.info(f"Fetching articles from {start_date_str} to {end_date_str}")
    
    # First, count the total number of articles in the date range
    count_result = supabase.table("source_articles") \
        .select("*", count="exact") \
        .gte("published", start_date_str) \
        .lte("published", end_date_str) \
        .execute()
    
    total_count = count_result.count
    log.info(f"Found {total_count} articles in the date range")
    
    if total_count == 0:
        log.warning(f"No articles found in database for the last {days} days.")
        return pd.DataFrame(columns=['id', 'date', 'title', 'text'])
    
    # Use pagination to fetch all articles (Supabase has a 1000 row limit)
    all_articles = []
    batch_size = 1000
    
    for offset in range(0, total_count, batch_size):
        end_range = min(offset + batch_size - 1, total_count - 1)
        log.info(f"Fetching articles {offset} to {end_range} of {total_count}")
        
        batch_result = supabase.table("source_articles") \
            .select("*") \
            .gte("published", start_date_str) \
            .lte("published", end_date_str) \
            .range(offset, end_range) \
            .execute()
        
        all_articles.extend(batch_result.data)
    
    df = pd.DataFrame(all_articles)
    
    # Debug: Print available columns
    log.info(f"Available columns from Supabase: {list(df.columns)}")
    log.info(f"Number of rows returned: {len(df)}")
    if len(df) > 0:
        log.info(f"Sample row keys: {list(df.iloc[0].keys()) if len(df) > 0 else 'No data'}")
    
    # Handle empty DataFrame
    if len(df) == 0:
        log.warning(f"No articles found in database for the last {days} days.")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['id', 'date', 'title', 'text'])
    
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

def main_days(days: int):
    """Process articles from the last N days."""
    df = read_articles_from_days(days)
    
    if len(df) == 0:
        log.warning(f"No articles found for the last {days} days. Nothing to process.")
        return
    
    log.info(f"Processing {len(df)} articles from the last {days} days")
    proc_df = preprocess_articles(df)
    write_processed_articles_to_supabase_and_pinecone(proc_df)

def main_all(limit=None):
    """Process all articles (cold start)."""
    df = read_all_articles()
    
    # Apply limit if specified
    if limit:
        df = df.head(limit)
        log.info(f"Limited to {len(df)} articles for testing")
    
    proc_df = preprocess_articles(df)
    write_processed_articles_to_supabase_and_pinecone(proc_df)
