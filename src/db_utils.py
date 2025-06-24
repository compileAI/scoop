"""db_utils.py

Unified database utilities for the clustering pipeline.
Handles data loading from Supabase and Pinecone, and cluster storage operations.
"""

import os
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from pinecone import Pinecone
from supabase import create_client
from dotenv import load_dotenv

from data_types import (
    ProcessedArticlesDF, ClustersDF, EmbeddingsDict, ChunksDict, ChunkData,
    PINECONE_INDEX_NAME, PINECONE_NAMESPACE, 
    SUPABASE_ARTICLES_TABLE, SUPABASE_CHUNKS_TABLE,
    SUPABASE_CLUSTERS_TABLE, SUPABASE_ASSIGNMENTS_TABLE
)
from preprocess import tokenize_texts

load_dotenv()

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize clients at module level
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

def fetch_processed_articles_by_date(target_date: datetime, 
                                   limit: Optional[int] = None) -> ProcessedArticlesDF:
    """
    Fetch all processed articles for a specific date.
    
    Args:
        target_date: Date to fetch articles for (YYYY-MM-DD)
        limit: Optional limit on number of articles
    
    Returns:
        DataFrame with processed article structure
    """
    date_str = target_date.strftime('%Y-%m-%d')
    date_filter = f"date.gte.{date_str}T00:00:00Z,date.lt.{date_str}T23:59:59Z"
    
    return _fetch_and_assemble_articles(date_filter, limit)

def fetch_processed_articles_since_date(since_date: datetime, 
                                      limit: Optional[int] = None) -> ProcessedArticlesDF:
    """
    Fetch all processed articles since a specific date.
    
    Args:
        since_date: Date to fetch articles since
        limit: Optional limit on number of articles
    
    Returns:
        DataFrame with processed article structure
    """
    date_filter = f"date.gte.{since_date.isoformat()}"
    return _fetch_and_assemble_articles(date_filter, limit)

def _fetch_and_assemble_articles(date_filter: str, limit: Optional[int] = None) -> ProcessedArticlesDF:
    """Internal helper to fetch and assemble articles with date filtering."""
    # Step 1: Fetch articles from Supabase
    articles_df = fetch_articles_from_supabase(date_filter, limit)
    
    if len(articles_df) == 0:
        print(f"WARNING: No articles found for the given date range: {date_filter}")
        return pd.DataFrame()
    
    # Step 2: Get article IDs and fetch chunks
    article_ids = articles_df['source_article_id'].tolist()
    chunks_dict = fetch_chunks_for_articles(article_ids)
    
    if not chunks_dict:
        print("WARNING: No chunks found for any articles")
        return pd.DataFrame()
    
    # Step 3: Collect all chunk IDs and fetch embeddings
    all_chunk_ids = []
    for chunks in chunks_dict.values():
        all_chunk_ids.extend([chunk.chunk_id for chunk in chunks])
    
    embeddings_dict = fetch_embeddings_from_pinecone(all_chunk_ids)
    
    if not embeddings_dict:
        print("WARNING: No embeddings found for any chunks")
        return pd.DataFrame()
    
    # Step 4: Assemble final DataFrame
    return assemble_processed_articles(articles_df, chunks_dict, embeddings_dict)

def fetch_articles_from_supabase(date_filter: str, 
                                limit: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch articles with date filtering from cleaned_source_articles.
    
    Args:
        date_filter: Date filter string for Supabase query
        limit: Optional limit on number of articles
    
    Returns:
        DataFrame with article metadata
    """
    try:
        query = supabase.table(SUPABASE_ARTICLES_TABLE).select("*")
        
        # Apply date filter if provided
        if date_filter:
            # Parse date filter (simplified - assumes gte format)
            if "gte." in date_filter:
                date_value = date_filter.split("gte.")[1]
                if "," in date_value:
                    date_value = date_value.split(",")[0]
                query = query.gte("date", date_value)
        
        if limit:
            query = query.limit(limit)
        
        result = query.execute()
        df = pd.DataFrame(result.data)
        
        if len(df) == 0:
            print("WARNING: No articles found in database")
            return pd.DataFrame()
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='ISO8601')
        
        print(f"INFO: Fetched {len(df)} articles from database")
        return df
        
    except Exception as e:
        print(f"ERROR: Error fetching articles from database: {e}")
        return pd.DataFrame()

def fetch_chunks_for_articles(article_ids: List[str], 
                            batch_size: int = 128) -> ChunksDict:
    """
    Fetch chunks for given article IDs (batched).
    
    Args:
        article_ids: List of article IDs to fetch chunks for
        batch_size: Batch size for database queries
    
    Returns:
        Dictionary mapping article_id to list of ChunkData
    """
    chunks_dict = {}
    
    # Process articles in batches
    for i in range(0, len(article_ids), batch_size):
        batch_ids = article_ids[i:i + batch_size]
        
        try:
            # Fetch chunks from database for this batch
            chunks_result = supabase.table(SUPABASE_CHUNKS_TABLE)\
                .select("chunk_id, text, source_article_id")\
                .in_("source_article_id", batch_ids)\
                .execute()
            
            if not chunks_result.data:
                print(f"WARNING: No chunks found for batch: {batch_ids}")
                continue
            
            # Group chunks by article for this batch
            for chunk_data in chunks_result.data:
                article_id = chunk_data['source_article_id']
                if article_id not in chunks_dict:
                    chunks_dict[article_id] = []
                
                chunk = ChunkData(
                    chunk_id=chunk_data['chunk_id'],
                    text=chunk_data['text'],
                    source_article_id=article_id
                )
                chunks_dict[article_id].append(chunk)
                
        except Exception as e:
            print(f"ERROR: Error fetching chunks for batch {i//batch_size + 1}: {e}")
            continue
        
        # Small delay to be respectful to the API
        time.sleep(0.1)
    
    print(f"INFO: Fetched chunks for {len(chunks_dict)} articles")
    return chunks_dict

def fetch_embeddings_from_pinecone(chunk_ids: List[int], 
                                 batch_size: int = 1000) -> EmbeddingsDict:
    """
    Fetch embeddings from Pinecone for given chunk IDs (batched).
    
    Args:
        chunk_ids: List of chunk IDs to fetch embeddings for
        batch_size: Batch size for Pinecone queries
    
    Returns:
        Dictionary mapping chunk_id to embedding array
    """
    embeddings = {}
    
    for i in range(0, len(chunk_ids), batch_size):
        batch_ids = chunk_ids[i:i + batch_size]
        
        try:
            # Convert chunk_ids to strings as Pinecone expects string IDs
            string_ids = [str(cid) for cid in batch_ids]
            
            result = pinecone_index.fetch(
                ids=string_ids,
                namespace=PINECONE_NAMESPACE
            )
            
            # Handle the Pinecone response format
            if hasattr(result, 'vectors') and result.vectors:
                for chunk_id, data in result.vectors.items():
                    embeddings[int(chunk_id)] = np.array(data.values)
            elif isinstance(result, dict) and 'vectors' in result:
                for chunk_id, data in result['vectors'].items():
                    embeddings[int(chunk_id)] = np.array(data['values'])
                
        except Exception as e:
            print(f"ERROR: Error fetching embeddings for batch {batch_ids}: {e}")
            
        # Rate limiting
        time.sleep(0.1)
    
    print(f"INFO: Fetched embeddings for {len(embeddings)} chunks")
    return embeddings

def assemble_processed_articles(articles_df: pd.DataFrame, 
                              chunks_dict: ChunksDict, 
                              embeddings_dict: EmbeddingsDict) -> ProcessedArticlesDF:
    """
    Assemble the final processed articles DataFrame.
    
    Args:
        articles_df: DataFrame with article metadata
        chunks_dict: Dictionary of chunks by article ID
        embeddings_dict: Dictionary of embeddings by chunk ID
    
    Returns:
        ProcessedArticlesDF with complete article structure
    """
    reconstructed_articles = []
    
    for _, article in articles_df.iterrows():
        article_id = article['source_article_id']
        
        # Skip articles without chunks
        if article_id not in chunks_dict:
            print(f"WARNING: No chunks found for article {article_id}")
            continue
            
        chunks = chunks_dict[article_id]
        
        # Extract chunk texts and embeddings
        chunk_texts = []
        chunk_embeddings = []
        
        for chunk in chunks:
            if chunk.chunk_id in embeddings_dict:
                chunk_texts.append(chunk.text)
                chunk_embeddings.append(embeddings_dict[chunk.chunk_id])
            else:
                print(f"WARNING: No embedding found for chunk {chunk.chunk_id}")
        
        # Skip articles without embeddings
        if not chunk_embeddings:
            print(f"WARNING: No embeddings found for article {article_id}")
            continue
        
        # Create sentences list (treating chunks as sentences)
        # Add title as first sentence if available
        sentences = []
        if pd.notna(article.get('title', '')):
            sentences.append(article['title'])
        sentences.extend(chunk_texts)
        
        # Create sentence embeddings (title gets average of chunk embeddings)
        sentence_embds = []
        if pd.notna(article.get('title', '')):
            # Use average of chunk embeddings for title
            title_embedding = np.mean(chunk_embeddings, axis=0)
            sentence_embds.append(title_embedding)
        sentence_embds.extend(chunk_embeddings)
        
        # Tokenize sentences
        sentence_tokens = tokenize_texts(sentences)
        
        reconstructed_article = {
            'id': article_id,
            'source_article_id': article_id,
            'date': article.get('date', article.get('created_at')),
            'title': article.get('title', ''),
            'text': article.get('text', ''),
            'sentences': sentences,
            'sentence_counts': len(sentences),
            'sentence_tokens': sentence_tokens,
            'sentence_embds': sentence_embds
        }
        
        reconstructed_articles.append(reconstructed_article)
    
    if len(reconstructed_articles) == 0:
        print("ERROR: No articles successfully reconstructed")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(reconstructed_articles)
    print(f"INFO: Successfully reconstructed {len(result_df)} articles")
    return result_df

# =============================================================================
# CLUSTER STORAGE FUNCTIONS
# =============================================================================

def save_cluster(cluster_center: np.ndarray, 
                keywords: List[str], 
                article_ids: List[str],
                date_range_start: datetime,
                date_range_end: datetime) -> int:
    """
    Save a new cluster and return its ID.
    
    Args:
        cluster_center: Cluster center embedding
        keywords: List of keywords for the cluster
        article_ids: List of article IDs in the cluster
        date_range_start: Start date of articles in cluster
        date_range_end: End date of articles in cluster
    
    Returns:
        Cluster ID of the saved cluster
    """
    try:
        # Insert cluster metadata
        cluster_data = {
            "cluster_center": cluster_center.tolist(),
            "cluster_size": len(article_ids),
            "keywords": keywords,
            "date_range_start": date_range_start.isoformat(),
            "date_range_end": date_range_end.isoformat(),
            "status": "active"
        }
        
        result = supabase.table(SUPABASE_CLUSTERS_TABLE).insert(cluster_data).execute()
        cluster_id = result.data[0]["cluster_id"]
        
        # Save article assignments
        assign_articles_to_cluster(cluster_id, article_ids, assignment_type="initial")
        
        return cluster_id
        
    except Exception as e:
        print(f"ERROR: Error saving cluster: {e}")
        raise

def load_active_clusters() -> ClustersDF:
    """
    Load all active clusters from database.
    
    Returns:
        DataFrame with cluster information
    """
    try:
        result = supabase.table(SUPABASE_CLUSTERS_TABLE)\
            .select("*")\
            .eq("status", "active")\
            .execute()
        
        df = pd.DataFrame(result.data)
        
        if len(df) > 0:
            # Convert cluster_center back to numpy arrays
            df["cluster_center"] = df["cluster_center"].apply(np.array)
            df["date_range_start"] = pd.to_datetime(df["date_range_start"])
            df["date_range_end"] = pd.to_datetime(df["date_range_end"])

        return df
        
    except Exception as e:
        print(f"ERROR: Error loading active clusters: {e}")
        return pd.DataFrame()

def assign_articles_to_cluster(cluster_id: int, 
                             article_ids: List[str], 
                             similarities: Optional[List[float]] = None,
                             assignment_type: str = "assigned") -> bool:
    """
    Assign multiple articles to a cluster.
    
    Args:
        cluster_id: ID of the cluster to assign to
        article_ids: List of article IDs to assign
        similarities: Optional list of similarity scores
        assignment_type: Type of assignment ("assigned", "initial", etc.)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        assignments = []
        for i, article_id in enumerate(article_ids):
            similarity = similarities[i] if similarities else 0.0
            assignments.append({
                "source_article_id": article_id,
                "cluster_id": cluster_id,
                "similarity_score": similarity,
                "assignment_type": assignment_type
            })
        
        batch_size = 100
        for i in range(0, len(assignments), batch_size):
            batch = assignments[i:i + batch_size]
            supabase.table(SUPABASE_ASSIGNMENTS_TABLE).insert(batch).execute()
        
        # Update cluster size
        _update_cluster_size(cluster_id)
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error assigning articles to cluster: {e}")
        return False

def update_cluster_center(cluster_id: int, 
                        new_center: np.ndarray,
                        keywords: Optional[List[str]] = None) -> bool:
    """
    Update cluster center and optionally keywords.
    
    Args:
        cluster_id: ID of the cluster to update
        new_center: New cluster center embedding
        keywords: Optional new keywords list
    
    Returns:
        True if successful, False otherwise
    """
    try:
        update_data = {
            "cluster_center": new_center.tolist(),
            "last_updated": datetime.now().isoformat()
        }
        
        if keywords:
            update_data["keywords"] = keywords
        
        supabase.table(SUPABASE_CLUSTERS_TABLE)\
            .update(update_data)\
            .eq("cluster_id", cluster_id)\
            .execute()
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error updating cluster center: {e}")
        return False

def get_article_assignments(article_id: str) -> pd.DataFrame:
    """
    Get cluster assignments for a specific article.
    
    Args:
        article_id: ID of the article to get assignments for
    
    Returns:
        DataFrame with assignment information
    """
    try:
        result = supabase.table(SUPABASE_ASSIGNMENTS_TABLE)\
            .select("*")\
            .eq("source_article_id", article_id)\
            .execute()
        
        return pd.DataFrame(result.data)
        
    except Exception as e:
        print(f"ERROR: Error getting article assignments: {e}")
        return pd.DataFrame()

def get_cluster_articles(cluster_id: int) -> List[str]:
    """
    Get all article IDs assigned to a cluster.
    
    Args:
        cluster_id: ID of the cluster
    
    Returns:
        List of article IDs
    """
    try:
        result = supabase.table(SUPABASE_ASSIGNMENTS_TABLE)\
            .select("source_article_id")\
            .eq("cluster_id", cluster_id)\
            .execute()
        
        return [row["source_article_id"] for row in result.data]
        
    except Exception as e:
        print(f"ERROR: Error getting cluster articles: {e}")
        return []

def update_cluster_date_range(cluster_id: int) -> bool:
    """
    Update cluster date range based on the published dates of all assigned articles.
    
    Args:
        cluster_id: ID of the cluster to update
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get all article IDs assigned to this cluster
        article_ids = get_cluster_articles(cluster_id)
        
        if not article_ids:
            print(f"WARNING: No articles found for cluster {cluster_id}")
            return True  # Not an error, just empty cluster
        
        # Query source_articles table to get published dates
        # Use batch processing for large clusters
        batch_size = 100
        all_dates = []
        
        for i in range(0, len(article_ids), batch_size):
            batch_ids = article_ids[i:i + batch_size]
            
            result = supabase.table('source_articles')\
                .select("published")\
                .in_("id", batch_ids)\
                .execute()
            
            if result.data:
                batch_dates = [row['published'] for row in result.data if row['published']]
                all_dates.extend(batch_dates)
        
        if not all_dates:
            print(f"WARNING: No published dates found for cluster {cluster_id} articles")
            return True  # Not an error, just no dates
        
        # Convert to datetime objects and find min/max
        from datetime import datetime
        parsed_dates = []
        for date_str in all_dates:
            try:
                if isinstance(date_str, str):
                    # Parse ISO format datetime string
                    parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    parsed_dates.append(parsed_date)
                elif hasattr(date_str, 'isoformat'):  # Already datetime object
                    parsed_dates.append(date_str)
            except Exception as e:
                print(f"WARNING: Could not parse date {date_str}: {e}")
                continue
        
        if not parsed_dates:
            print(f"WARNING: No valid dates found for cluster {cluster_id}")
            return True
        
        date_range_start = min(parsed_dates)
        date_range_end = max(parsed_dates)
        
        # Update cluster metadata
        update_data = {
            "date_range_start": date_range_start.isoformat(),
            "date_range_end": date_range_end.isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        supabase.table(SUPABASE_CLUSTERS_TABLE)\
            .update(update_data)\
            .eq("cluster_id", cluster_id)\
            .execute()
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error updating cluster date range for cluster {cluster_id}: {e}")
        return False

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _update_cluster_size(cluster_id: int) -> bool:
    """Update cluster size count."""
    try:
        # Count current assignments
        result = supabase.table(SUPABASE_ASSIGNMENTS_TABLE)\
            .select("*", count="exact")\
            .eq("cluster_id", cluster_id)\
            .execute()
        
        new_size = result.count
        
        # Update cluster metadata
        supabase.table(SUPABASE_CLUSTERS_TABLE)\
            .update({"cluster_size": new_size, "last_updated": datetime.now().isoformat()})\
            .eq("cluster_id", cluster_id)\
            .execute()
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error updating cluster size: {e}")
        return False


# =============================================================================
# CLUSTERING RESULTS PROCESSING FUNCTIONS
# =============================================================================

def extract_cluster_metadata(all_window: pd.DataFrame, cluster_keywords_df: pd.DataFrame) -> List[dict]:
    """
    Extract cluster metadata from clustering results.
    
    Args:
        all_window: DataFrame with all articles and their cluster assignments
        cluster_keywords_df: DataFrame with cluster keywords
    
    Returns:
        List of cluster metadata dictionaries
    """
    cluster_metadata = []
    
    # Get all valid clusters (not outliers)
    clustered_articles = all_window[all_window['cluster'] >= 0]
    
    if len(clustered_articles) == 0:
        return cluster_metadata
    
    # Process each unique cluster
    for cluster_id in sorted(clustered_articles['cluster'].unique()):
        cluster_articles = clustered_articles[clustered_articles['cluster'] == cluster_id]
        
        # Calculate cluster center (mean of article embeddings)
        embeddings = np.vstack(cluster_articles['embedding'].values)
        cluster_center = np.mean(embeddings, axis=0)
        
        # Get date range
        dates = pd.to_datetime(cluster_articles['date'])
        date_range_start = dates.min()
        date_range_end = dates.max()
        
        # Get keywords (try to get from last window that has this cluster)
        keywords = []
        if len(cluster_keywords_df) > 0 and cluster_id in cluster_keywords_df.columns:
            # Get the last non-empty keywords entry for this cluster
            for idx in reversed(cluster_keywords_df.index):
                if cluster_id in cluster_keywords_df.columns:
                    cell_value = cluster_keywords_df.at[idx, cluster_id]
                    # Handle different types of cell values
                    try:
                        # Check if it's a list/array
                        if isinstance(cell_value, (list, np.ndarray)):
                            if len(cell_value) > 0:
                                keywords = cell_value
                                break
                        # Check if it's a string
                        elif isinstance(cell_value, str) and cell_value.strip():
                            keywords = [cell_value]  # Convert to list for consistency
                            break
                        # Check if it's not NaN/None
                        elif cell_value is not None and not pd.isna(cell_value):
                            keywords = [str(cell_value)]
                            break
                    except Exception:
                        # Skip problematic values
                        continue
        
        # Ensure keywords is a list
        if not isinstance(keywords, list):
            keywords = []
        
        cluster_metadata.append({
            'cluster_center': cluster_center,
            'cluster_size': len(cluster_articles),
            'keywords': keywords,
            'date_range_start': date_range_start,
            'date_range_end': date_range_end,
            'status': 'active',
            'similarity_threshold': 0.7,  # Default threshold
            'original_cluster_id': cluster_id  # Keep track of original ID for mapping
        })
    
    return cluster_metadata


def extract_article_assignments(all_window: pd.DataFrame) -> List[dict]:
    """
    Extract article-cluster assignments from clustering results.
    
    Args:
        all_window: DataFrame with all articles and their cluster assignments
    
    Returns:
        List of article assignment dictionaries
    """
    assignments = []
    
    for _, article in all_window.iterrows():
        # Determine assignment details
        cluster_id = int(article['cluster']) if article['cluster'] >= 0 else None
        similarity_score = float(article['sim']) if pd.notna(article['sim']) else 0.0
        assignment_type = 'outlier' if cluster_id is None else 'initial'
        
        # Get article ID (try different possible column names)
        article_id = None
        for col in ['id', 'source_article_id', 'article_id']:
            if col in article.index and pd.notna(article[col]):
                article_id = str(article[col])
                break
        
        if article_id is None:
            print(f"WARNING: Could not find article ID for article in row")
            continue
        
        assignments.append({
            'source_article_id': article_id,
            'original_cluster_id': cluster_id,  # Will be mapped to DB cluster_id later
            'similarity_score': similarity_score,
            'assignment_type': assignment_type
        })
    
    return assignments


def save_clustering_results_to_db(all_window: pd.DataFrame, 
                                cluster_keywords_df: pd.DataFrame) -> bool:
    """
    Save complete clustering results to database.
    
    Args:
        all_window: DataFrame with all articles and their cluster assignments
        cluster_keywords_df: DataFrame with cluster keywords
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print("📊 Extracting cluster metadata...")
        cluster_metadata = extract_cluster_metadata(all_window, cluster_keywords_df)
        
        print("📊 Extracting article assignments...")
        assignments = extract_article_assignments(all_window)
        
        if len(cluster_metadata) == 0:
            print("WARNING: No clusters found to save")
            return True
        
        print(f"💾 Saving {len(cluster_metadata)} clusters to database...")
        
        # Step 1: Save clusters and get ID mapping
        cluster_id_mapping = {}  # original_cluster_id -> db_cluster_id
        
        for cluster_meta in cluster_metadata:
            original_id = cluster_meta.pop('original_cluster_id')
            
            # Convert numpy array to list for JSON serialization
            cluster_meta['cluster_center'] = cluster_meta['cluster_center'].tolist()
            
            # Convert datetime to ISO string
            cluster_meta['date_range_start'] = cluster_meta['date_range_start'].isoformat()
            cluster_meta['date_range_end'] = cluster_meta['date_range_end'].isoformat()
            
            # Insert cluster
            result = supabase.table(SUPABASE_CLUSTERS_TABLE).insert(cluster_meta).execute()
            db_cluster_id = result.data[0]["cluster_id"]
            cluster_id_mapping[original_id] = db_cluster_id
            
            print(f"   ✅ Cluster {original_id} saved as DB cluster {db_cluster_id}")
        
        # Step 2: Save article assignments
        print(f"💾 Saving {len(assignments)} article assignments...")
        
        # Update assignments with correct cluster IDs
        db_assignments = []
        for assignment in assignments:
            original_cluster_id = assignment.pop('original_cluster_id')
            if original_cluster_id is not None:
                assignment['cluster_id'] = cluster_id_mapping[original_cluster_id]
            else:
                assignment['cluster_id'] = None  # Outlier articles
            db_assignments.append(assignment)
        
        # Batch insert assignments
        batch_size = 100
        for i in range(0, len(db_assignments), batch_size):
            batch = db_assignments[i:i + batch_size]
            supabase.table(SUPABASE_ASSIGNMENTS_TABLE).insert(batch).execute()
            print(f"   ✅ Saved assignments batch {i//batch_size + 1}/{(len(db_assignments)-1)//batch_size + 1}")
        
        print("🎉 Successfully saved all clustering results to database!")
        
        # Print summary
        clustered_count = len([a for a in assignments if a.get('assignment_type') != 'outlier'])
        outlier_count = len([a for a in assignments if a.get('assignment_type') == 'outlier'])
        
        print("=" * 60)
        print("📊 Database Save Summary:")
        print(f"   • Clusters saved: {len(cluster_metadata)}")
        print(f"   • Articles clustered: {clustered_count}")
        print(f"   • Outlier articles: {outlier_count}")
        print(f"   • Total articles processed: {len(assignments)}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving clustering results: {e}")
        import traceback
        traceback.print_exc()
        return False
