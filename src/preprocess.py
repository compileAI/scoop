from __future__ import annotations

from datetime import datetime, timezone
import os
import time
from typing import List, Iterable

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
from gemini_api_manager import get_gemini_manager, DailyQuotaExhausted

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Gemini API manager with key rotation
gemini_manager = get_gemini_manager()
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("scraped-sources-gemini")

# spaCy pipeline – blank model + sentencizer (removed lemmatizer to avoid lookup table issues)
def build_nlp() -> spacy.language.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    # Note: Removed lemmatizer as it requires lookup tables not available in blank model
    print(f"spaCy pipeline ready: {nlp.pipe_names}")
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
    print(f"Dataframe shape before processing: {df.shape}")
    
    # Handle empty DataFrame
    if len(df) == 0:
        print("Empty DataFrame provided, returning empty processed DataFrame")
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
        print(f"Cleaned {title_none_count} None titles and {text_none_count} None text values")

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

    # Embedding with automatic key rotation (handled by GeminiApiManager)
    embeddings = []
    max_batch_size = 100  # Google API limit
    
    for i, batch in enumerate(tqdm(sentences, desc="Embedding sentences", total=len(sentences))):
        if len(batch) <= max_batch_size:
            # Batch fits within API limit - manager handles rate limits automatically
            batch_embeddings = gemini_manager.embed_content(
                model="gemini-embedding-001",
                contents=batch,
                task_type="CLUSTERING",
                output_dimensionality=768
            )
            # Extract the actual embedding values correctly
            actual_embeddings = [emb.values for emb in batch_embeddings.embeddings]
            embeddings.append(actual_embeddings)
        else:
            # Split large batches into smaller chunks
            article_embeddings = []
            for chunk_start in range(0, len(batch), max_batch_size):
                chunk = batch[chunk_start:chunk_start + max_batch_size]
                chunk_embeddings = gemini_manager.embed_content(
                    model="gemini-embedding-001",
                    contents=chunk,
                    task_type="CLUSTERING",
                    output_dimensionality=768
                )
                actual_embeddings = [emb.values for emb in chunk_embeddings.embeddings]
                article_embeddings.extend(actual_embeddings)
            embeddings.append(article_embeddings)

    # assemble dataframe using the cleaned data
    out = df_clean.copy()
    out["sentences"] = sentences
    out["sentence_counts"] = sentence_counts
    out["sentence_tokens"] = sentence_tokens
    out["sentence_embds"] = embeddings

    print(f"Completed preprocessing - final shape: {out.shape}")
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
    print(f"Available columns from Supabase: {list(df.columns)}")
    print(f"Number of rows returned: {len(df)}")
    if len(df) > 0:
        print(f"Sample row keys: {list(df.iloc[0].keys()) if len(df) > 0 else 'No data'}")
    
    # Handle empty DataFrame
    if len(df) == 0:
        print("No articles found for today. Consider using --cold_start to process all articles.")
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
    
    print(f"Found {total_count} total articles in database")
    
    if total_count == 0:
        print("No articles found in database.")
        return pd.DataFrame(columns=['id', 'date', 'title', 'text'])
    
    # Use pagination to fetch all articles (Supabase has a 1000 row limit)
    all_articles = []
    batch_size = 1000
    
    for offset in range(0, total_count, batch_size):
        end_range = min(offset + batch_size - 1, total_count - 1)
        print(f"Fetching articles {offset} to {end_range} of {total_count}")
        
        batch_result = supabase.table("source_articles") \
            .select("*") \
            .range(offset, end_range) \
            .execute()
        
        all_articles.extend(batch_result.data)
    
    df = pd.DataFrame(all_articles)
    
    # Debug: Print available columns
    print(f"Available columns from Supabase: {list(df.columns)}")
    print(f"Number of rows returned: {len(df)}")
    if len(df) > 0:
        print(f"Sample row keys: {list(df.iloc[0].keys()) if len(df) > 0 else 'No data'}")
    
    # Handle empty DataFrame
    if len(df) == 0:
        print("No articles found in database.")
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
            print(f"Creating Pinecone namespace '{namespace}'")
            # Create namespace by upserting a dummy vector with non-zero values and then deleting it
            dummy_vector = {
                'id': 'dummy_init_vector',
                'values': [0.1] * 768,  # Non-zero values for gemini-embedding-001 (768-dimensional)
                'metadata': {'init': True}
            }
            pinecone_index.upsert(vectors=[dummy_vector], namespace=namespace)
            # Delete the dummy vector
            pinecone_index.delete(ids=['dummy_init_vector'], namespace=namespace)
            print(f"Pinecone namespace '{namespace}' created successfully")
        else:
            print(f"Pinecone namespace '{namespace}' already exists")
    except Exception as e:
        print(f"Error ensuring Pinecone namespace exists: {e}")
        raise

def write_processed_articles_to_supabase_and_pinecone(proc_df: pd.DataFrame) -> None:
    """
    Write processed articles to Supabase tables and embeddings to Pinecone.
    
    Args:
        proc_df: DataFrame with columns: id, date, title, text, sentences, 
                sentence_counts, sentence_tokens, sentence_embds
    """
    print(f"Writing {len(proc_df)} processed articles to Supabase and Pinecone")
    
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
                'date': pd.Timestamp(row['date']).isoformat() if pd.notna(row['date']) else None
            }
            
            # Upsert into cleaned_source_articles (insert or update if exists)
            # Use on_conflict parameter to specify which column to use for conflict resolution
            article_result = supabase.table(SUPABASE_ARTICLES_TABLE).upsert(
                article_data, 
                on_conflict="source_article_id"
            ).execute()
            
            if not article_result.data:
                print(f"Failed to upsert article {idx}")
                continue
                
            source_article_id = row['id']  # Original string ID, now the unique identifier
            print(f"Upserted article with source_article_id: {source_article_id}")
            
            # Process each sentence as a chunk
            sentences = row['sentences']
            embeddings = row['sentence_embds']
            
            # First, delete any existing chunks for this article (in case we're reprocessing)
            existing_chunks = supabase.table(SUPABASE_CHUNKS_TABLE).select("chunk_id").eq('source_article_id', source_article_id).execute()
            if existing_chunks.data:
                # Delete old Pinecone vectors
                old_chunk_ids = [str(chunk['chunk_id']) for chunk in existing_chunks.data]
                pinecone_index.delete(ids=old_chunk_ids, namespace="source_article_clustering_chunks")
                print(f"Deleted {len(old_chunk_ids)} old Pinecone vectors for source_article_id: {source_article_id}")
            
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

                # Quick error checks
                if hasattr(chunk_result, 'error') and chunk_result.error:
                    print(f"❌ Supabase error: {chunk_result.error} for article {source_article_id}")
                    continue

                if not chunk_result.data or len(chunk_result.data) != len(chunk_data):
                    print(f"⚠️ Chunk insert failed: expected {len(chunk_data)}, got {len(chunk_result.data or [])} for article {source_article_id}")
                    continue
                
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
                        print(f"Upserted {len(pinecone_vectors)} vectors to Pinecone for source_article_id: {source_article_id}")
                else:
                    print(f"Failed to insert chunks for source_article_id: {source_article_id}")
            
        except Exception as e:
            print(f"Error processing article {idx}: {e}")
            continue
    
    print(f"Successfully wrote {len(proc_df)} articles and {total_chunks} chunks to database and Pinecone")

def read_unprocessed_articles(days: int = 3) -> pd.DataFrame:
    """
    Read articles from the last N days that haven't been preprocessed yet.
    Returns articles ordered by date descending (most recent first).
    
    Args:
        days: Number of days to look back (default: 3)
        
    Returns:
        DataFrame of unprocessed articles, sorted by date descending
    """
    print(f"Checking for unprocessed articles from the last {days} days")
    
    # Get articles from last N days
    recent_articles = read_articles_from_days(days)
    
    if len(recent_articles) == 0:
        print(f"No articles found in the last {days} days")
        return pd.DataFrame(columns=['id', 'date', 'title', 'text'])
    
    print(f"Found {len(recent_articles)} total articles from the last {days} days")
    
    # Get all processed article IDs
    try:
        processed_result = supabase.table(SUPABASE_ARTICLES_TABLE) \
            .select("source_article_id") \
            .execute()
        
        processed_ids = set(r['source_article_id'] for r in processed_result.data)
        print(f"Found {len(processed_ids)} already processed articles")
    except Exception as e:
        print(f"Error fetching processed articles: {e}")
        processed_ids = set()
    
    # Filter to unprocessed articles only
    unprocessed = recent_articles[~recent_articles['id'].isin(processed_ids)].copy()
    
    # Sort by date descending (most recent first)
    unprocessed['date'] = pd.to_datetime(unprocessed['date'], format='ISO8601')
    unprocessed = unprocessed.sort_values('date', ascending=False)
    
    print(f"Found {len(unprocessed)} unprocessed articles")
    
    return unprocessed


def read_unprocessed_articles(days: int = 3) -> pd.DataFrame:
    """
    Read articles from the last N days that haven't been preprocessed yet.
    Returns articles ordered by date descending (most recent first).
    
    Args:
        days: Number of days to look back (default: 3)
        
    Returns:
        DataFrame of unprocessed articles, sorted by date descending
    """
    print(f"Checking for unprocessed articles from the last {days} days")
    
    # Get articles from last N days
    recent_articles = read_articles_from_days(days)
    
    if len(recent_articles) == 0:
        print(f"No articles found in the last {days} days")
        return pd.DataFrame(columns=['id', 'date', 'title', 'text'])
    
    print(f"Found {len(recent_articles)} total articles from the last {days} days")
    
    # Get all processed article IDs
    try:
        processed_result = supabase.table(SUPABASE_ARTICLES_TABLE) \
            .select("source_article_id") \
            .execute()
        
        processed_ids = set(r['source_article_id'] for r in processed_result.data)
        print(f"Found {len(processed_ids)} already processed articles")
    except Exception as e:
        print(f"Error fetching processed articles: {e}")
        processed_ids = set()
    
    # Filter to unprocessed articles only
    unprocessed = recent_articles[~recent_articles['id'].isin(processed_ids)].copy()
    
    # Sort by date descending (most recent first)
    unprocessed['date'] = pd.to_datetime(unprocessed['date'], format='ISO8601')
    unprocessed = unprocessed.sort_values('date', ascending=False)
    
    print(f"Found {len(unprocessed)} unprocessed articles")
    
    return unprocessed


def main_daily():
    """
    Process unprocessed articles from the last 3 days, one at a time.
    Most recent articles are processed first.
    """
    print("=" * 80)
    print("Starting incremental daily preprocessing")
    print("=" * 80)
    
    # Get unprocessed articles from last 3 days
    df = read_unprocessed_articles(days=3)
    
    if len(df) == 0:
        print("✅ No unprocessed articles found. All caught up!")
        return
    
    print(f"📋 Processing {len(df)} unprocessed articles (most recent first)")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print("=" * 80)
    
    successful_count = 0
    failed_count = 0
    failed_ids = []
    
    for idx, (row_idx, row) in enumerate(df.iterrows(), 1):
        article_id = row['id']
        article_date = row['date']
        
        try:
            print(f"\n[{idx}/{len(df)}] Processing article {article_id} from {article_date}")
            
            # Process single article
            single_df = pd.DataFrame([row])
            proc_df = preprocess_articles(single_df, batch_size=64)
            
            # Write immediately to database
            write_processed_articles_to_supabase_and_pinecone(proc_df)
            
            successful_count += 1
            print(f"✅ Successfully processed article {article_id} ({successful_count}/{len(df)} completed)")
            
        except DailyQuotaExhausted as e:
            print("=" * 80)
            print("❌ DAILY QUOTA EXHAUSTED")
            print("=" * 80)
            print(f"Processed {successful_count} articles successfully before hitting quota")
            print(f"Remaining: {len(df) - idx} unprocessed articles")
            print("These will be automatically retried on the next run.")
            print(f"Error details: {str(e)[:200]}")
            print("=" * 80)
            break
            
        except Exception as e:
            failed_count += 1
            failed_ids.append(article_id)
            print(f"❌ Failed to process article {article_id}: {str(e)[:200]}")
            print(f"Continuing with next article... ({failed_count} failures so far)")
            continue
    
    # Final summary
    print("=" * 80)
    print("Daily preprocessing complete")
    print("=" * 80)
    print(f"✅ Successfully processed: {successful_count} articles")
    if failed_count > 0:
        print(f"❌ Failed: {failed_count} articles")
        print(f"Failed article IDs: {failed_ids}")
    print(f"📊 Total processed this run: {successful_count + failed_count} / {len(df)}")
    print("=" * 80)

def read_articles_from_days(days: int) -> pd.DataFrame:
    """Read articles from the last N days from Supabase."""
    from datetime import datetime, timedelta
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Format dates for Supabase query
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Fetching articles from {start_date_str} to {end_date_str}")
    
    # First, count the total number of articles in the date range
    count_result = supabase.table("source_articles") \
        .select("*", count="exact") \
        .gte("created_at", start_date_str) \
        .lte("created_at", end_date_str) \
        .execute()
    
    total_count = count_result.count
    print(f"Found {total_count} articles in the date range")
    
    if total_count == 0:
        print(f"No articles found in database for the last {days} days.")
        return pd.DataFrame(columns=['id', 'date', 'title', 'text'])
    
    # Use pagination to fetch all articles (Supabase has a 1000 row limit)
    all_articles = []
    batch_size = 1000
    
    for offset in range(0, total_count, batch_size):
        end_range = min(offset + batch_size - 1, total_count - 1)
        print(f"Fetching articles {offset} to {end_range} of {total_count}")
        
        batch_result = supabase.table("source_articles") \
            .select("*") \
            .gte("created_at", start_date_str) \
            .lte("created_at", end_date_str) \
            .range(offset, end_range) \
            .execute()
        
        all_articles.extend(batch_result.data)
    
    df = pd.DataFrame(all_articles)
    
    # Debug: Print available columns
    print(f"Available columns from Supabase: {list(df.columns)}")
    print(f"Number of rows returned: {len(df)}")
    if len(df) > 0:
        print(f"Sample row keys: {list(df.iloc[0].keys()) if len(df) > 0 else 'No data'}")
    
    # Handle empty DataFrame
    if len(df) == 0:
        print(f"No articles found in database for the last {days} days.")
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
        print(f"No articles found for the last {days} days. Nothing to process.")
        return
    
    print(f"Processing {len(df)} articles from the last {days} days")
    proc_df = preprocess_articles(df)
    write_processed_articles_to_supabase_and_pinecone(proc_df)

def main_all(limit=None):
    """Process all articles (cold start)."""
    df = read_all_articles()
    
    # Apply limit if specified
    if limit:
        df = df.head(limit)
        print(f"Limited to {len(df)} articles for testing")
    
    proc_df = preprocess_articles(df)
    write_processed_articles_to_supabase_and_pinecone(proc_df)
