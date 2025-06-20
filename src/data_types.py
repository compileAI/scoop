"""data_types.py

Type definitions and data classes for the clustering pipeline database utilities.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

@dataclass
class ProcessedArticle:
    """Represents a processed article with all required fields."""
    id: str
    date: datetime
    title: str
    text: str
    sentences: List[str]
    sentence_counts: int
    sentence_tokens: List[List[str]]
    sentence_embds: List[np.ndarray]

@dataclass
class ClusterInfo:
    """Represents cluster metadata."""
    cluster_id: int
    center: np.ndarray
    keywords: List[str]
    article_ids: List[str]
    size: int
    date_range_start: datetime
    date_range_end: datetime

@dataclass
class ChunkData:
    """Represents a text chunk with metadata."""
    chunk_id: int
    text: str
    source_article_id: str

# Type aliases for better readability
ProcessedArticlesDF = pd.DataFrame  # DataFrame with ProcessedArticle structure
ClustersDF = pd.DataFrame          # DataFrame with ClusterInfo structure
EmbeddingsDict = Dict[int, np.ndarray]  # Mapping chunk_id -> embedding
ChunksDict = Dict[str, List[ChunkData]]  # Mapping article_id -> list of chunks

# Constants
PINECONE_INDEX_NAME = "scraped-sources-gemini"
PINECONE_NAMESPACE = "chunks"
SUPABASE_ARTICLES_TABLE = "cleaned_source_articles"
SUPABASE_CHUNKS_TABLE = "cleaned_source_article_chunks"
SUPABASE_CLUSTERS_TABLE = "clusters"
SUPABASE_ASSIGNMENTS_TABLE = "article_clusters" 