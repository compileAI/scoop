#!/usr/bin/env python3
"""
daily_clustering.py

Incremental clustering script for new articles against existing clusters.
Processes new articles within the context of a broader time window for proper embeddings.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from clustering import get_article_embedding, assign_to_clusters, cluster_outliers, get_cluster_theme, read_dataset
from db_utils import (
    load_active_clusters,
    save_cluster,
    assign_articles_to_cluster,
    update_cluster_center,
    update_cluster_date_range,
    get_cluster_articles
)

def load_and_prepare_data(days_back: int, window_size: int, verbose: bool):
    """Load new articles and context window for proper embedding computation."""
    
    # Calculate date ranges
    now = datetime.now()
    new_articles_since = (now - timedelta(days=days_back)).replace(hour=0, minute=0, second=0, microsecond=0)
    context_since = (now - timedelta(days=window_size)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    if verbose:
        print(f"📅 Loading new articles since: {new_articles_since.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📅 Loading context window since: {context_since.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load all articles using read_dataset (includes preprocessing)
    to_date = now.strftime('%Y-%m-%d')
    from_date = context_since.strftime('%Y-%m-%d')
    
    context_articles, all_vocab = read_dataset(
        start_date=from_date,
        end_date=to_date,
        verbose=verbose
    )
    
    if len(context_articles) == 0:
        print("❌ No context articles found")
        return None, None, None, None
    
    # Extract new articles from context
    new_articles = context_articles[context_articles['date'] >= new_articles_since].copy()
    
    # Load existing clusters
    existing_clusters = load_active_clusters()
    
    if verbose:
        print(f"✅ Context articles: {len(context_articles)}")
        print(f"✅ New articles: {len(new_articles)}")
        print(f"✅ Existing clusters: {len(existing_clusters)}")
    
    return context_articles, new_articles, existing_clusters, all_vocab

def convert_clusters_to_simulate_format(existing_clusters, context_articles, window_size: int, verbose: bool):
    """Convert database cluster format to the format expected by clustering functions."""
    
    if len(existing_clusters) == 0:
        if verbose:
            print("⚠️  No existing clusters found - this will initialize new clusters")
        return [], [], [], {}, {}, {}, {}

    cluster_centers = []
    cluster_emb_sum_dics = []  # List of dictionaries (not dict)
    cluster_tf_sum_dics = []   # List of dictionaries (not dict)
    cluster_topN_indices = {}
    cluster_topN_scores = {}
    cluster_topN_probs = {}
    
    # Create mapping from database cluster IDs to array indices
    db_cluster_ids = existing_clusters['cluster_id'].tolist()
    cluster_id_mapping = {db_id: idx for idx, db_id in enumerate(db_cluster_ids)}
    
    now = datetime.now()
    
    def ensure_numpy_array(center):
        """Convert cluster center to numpy array."""
        if isinstance(center, str):
            try:
                # First try using json.loads as it's more robust for array strings
                import json
                return np.array(json.loads(center))
            except:
                try:
                    # If that fails, try literal_eval for safety
                    import ast
                    return np.array(ast.literal_eval(center))
                except:
                    try:
                        # Last resort: use eval (less safe but sometimes necessary)
                        return np.array(eval(center))
                    except:
                        # Final fallback: try to parse as space-separated numbers
                        if center.startswith('[') and center.endswith(']'):
                            center_clean = center.strip('[]').replace('\n', ' ').replace('  ', ' ')
                            return np.fromstring(center_clean, sep=' ')
                        else:
                            return np.fromstring(center, sep=' ')
        elif isinstance(center, list):
            return np.array(center)
        elif isinstance(center, np.ndarray):
            return center
        else:
            return np.array(center)
    
    for idx, (_, cluster) in enumerate(existing_clusters.iterrows()):
        db_cluster_id = cluster['cluster_id']
        array_idx = cluster_id_mapping[db_cluster_id]
        
        # Store cluster center (ensure it's numpy array)
        cluster_center = ensure_numpy_array(cluster['cluster_center'])
        if verbose and idx < 3:  # Show first 3 for debugging
            print(f"   • Cluster {db_cluster_id}: center type {type(cluster_center)}, shape {getattr(cluster_center, 'shape', 'No shape')}")
        cluster_centers.append(cluster_center)
        
        # Initialize empty dictionaries and append to lists
        cluster_emb_sum_dics.append({})
        cluster_tf_sum_dics.append({})
        
        # Get articles assigned to this cluster within the window
        cluster_article_ids = get_cluster_articles(db_cluster_id)
        cluster_articles = context_articles[
            context_articles['source_article_id'].isin(cluster_article_ids)
        ]
        
        # Build embedding and TF sums by date for time-aware clustering
        for date in cluster_articles['date'].dt.date.unique():
            date_pd = pd.Timestamp(date)
            delta = (now.date() - date).days
            if delta >= window_size:
                continue
                
            articles_on_date = cluster_articles[cluster_articles['date'].dt.date == date]
            if len(articles_on_date) > 0:
                # Sum embeddings for this date
                emb_sum = np.sum([emb for emb in articles_on_date['embedding'].values], axis=0)
                count = len(articles_on_date)
                cluster_emb_sum_dics[array_idx][date_pd] = [emb_sum, count]
                
                # Sum TF vectors for this date
                if len(articles_on_date) > 0 and 'article_TF' in articles_on_date.columns:
                    tf_sum = sum(articles_on_date['article_TF'].values)
                    cluster_tf_sum_dics[array_idx][date_pd] = tf_sum

    if verbose:
        print(f"🔄 Converted {len(cluster_centers)} clusters to simulate format")
        print(f"   • Cluster ID mapping: {cluster_id_mapping}")

    return (cluster_centers, cluster_emb_sum_dics, cluster_tf_sum_dics, 
            cluster_topN_indices, cluster_topN_scores, cluster_topN_probs, cluster_id_mapping)

def save_clustering_results(window, cluster_centers, original_cluster_count, verbose: bool):
    """Save clustering results to database."""
    
    try:
        # Track assignments and updates
        assignments_saved = 0
        clusters_updated = 0
        new_clusters_created = 0
        
        # Get unique cluster IDs
        assigned_clusters = window[window['cluster'] >= 0]['cluster'].unique()
        
        for cluster_id in assigned_clusters:
            cluster_articles = window[window['cluster'] == cluster_id]
            article_ids = cluster_articles['source_article_id'].tolist()
            similarities = cluster_articles['sim'].tolist()
            
            if cluster_id >= original_cluster_count:
                # This is a new cluster - save it
                cluster_center = cluster_centers[cluster_id]
                keywords = []  # Will be populated later by get_cluster_theme
                date_range_start = cluster_articles['date'].min()
                date_range_end = cluster_articles['date'].max()
                
                saved_cluster_id = save_cluster(
                    cluster_center, keywords, article_ids, 
                    date_range_start, date_range_end
                )
                new_clusters_created += 1
                if verbose:
                    print(f"💾 Created new cluster {saved_cluster_id} with {len(article_ids)} articles")
            else:
                # Update existing cluster
                cluster_center = cluster_centers[cluster_id]
                update_cluster_center(cluster_id, cluster_center)
                clusters_updated += 1
                
                # Assign new articles to existing cluster
                assign_articles_to_cluster(
                    cluster_id, article_ids, similarities, assignment_type="daily"
                )
                assignments_saved += len(article_ids)
                
                # Update cluster date range based on all articles (including new ones)
                update_cluster_date_range(cluster_id)
                
                if verbose:
                    print(f"🔄 Updated cluster {cluster_id} with {len(article_ids)} new articles (date range updated)")
        
        # Summary
        outliers = len(window[window['cluster'] == -1])
        if verbose:
            print("=" * 60)
            print("💾 Database save summary:")
            print(f"   • Article assignments saved: {assignments_saved}")
            print(f"   • Existing clusters updated: {clusters_updated}")
            print(f"   • New clusters created: {new_clusters_created}")
            print(f"   • Outlier articles: {outliers}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving clustering results: {e}")
        return False

def main():
    """Main entry point for daily incremental clustering."""
    
    # Hard-coded clustering parameters (matching initial_batch_clustering.py)
    days_back = 0           # Process articles from last N days
    window_size = 14        # Context window for embeddings (days)
    min_articles = 6        # Minimum articles per new cluster
    N = 10                  # Top N keywords
    T = 4                   # Similarity threshold parameter
    keyword_score = "tfidf" # Keyword scoring method
    verbose = True          # Verbose output
    time_aware = True       # Time-aware clustering
    theme_aware = True      # Theme-aware clustering (disabled - no theme data from DB)
    
    print("🚀 Starting daily incremental clustering")
    print("=" * 60)
    print(f"📊 Parameters:")
    print(f"   • Days back: {days_back}")
    print(f"   • Context window: {window_size} days")
    print(f"   • Min articles per cluster: {min_articles}")
    print(f"   • T parameter: {T}")
    print(f"   • Top N keywords: {N}")
    print(f"   • Keyword scoring: {keyword_score}")
    print(f"   • Time aware: {time_aware}")
    print(f"   • Theme aware: {theme_aware}")
    print("=" * 60)
    
    try:
        # 1. Load data
        context_articles, new_articles, existing_clusters, all_vocab = load_and_prepare_data(
            days_back, window_size, verbose
        )
        
        if context_articles is None:
            print("❌ Failed to load data")
            return False
            
        if len(new_articles) == 0:
            print("📭 No new articles to process")
            return True
        
        # 2. Build article_df_slides for context (needed for get_article_embedding)
        # This is a simplified version - in practice you might need more sophisticated handling
        article_df_slides = []
        if len(context_articles) > 0 and 'article_TF' in context_articles.columns:
            from scipy.sparse import vstack
            try:
                article_TFs = vstack(context_articles["article_TF"])
                article_df_slide = np.bincount(
                    article_TFs.indices, minlength=article_TFs.shape[1]
                ).reshape(1, -1)
                article_df_slides.append(article_df_slide)
            except:
                article_df_slides = []
        
        # 3. Process embeddings with context FIRST
        if verbose:
            print("🔮 Computing embeddings for all context articles...")
        
        # Set all new articles as unassigned initially
        new_articles['cluster'] = -1
        new_articles['sim'] = 0.0
        
        # Compute embeddings for the new articles (slide) using context
        new_articles, _ = get_article_embedding(
            slide=new_articles,
            window=context_articles,
            article_df_slides=article_df_slides,
            time_aware=time_aware,
            theme_aware=theme_aware,
            keyword_score=keyword_score,
            N=N
        )
        
        # Also compute embeddings for context articles if they don't have them
        if 'embedding' not in context_articles.columns:
            if verbose:
                print("🔮 Computing embeddings for context articles...")
            context_articles, _ = get_article_embedding(
                slide=context_articles,
                window=context_articles,
                article_df_slides=article_df_slides,
                time_aware=time_aware,
                theme_aware=theme_aware,
                keyword_score=keyword_score,
                N=N
            )
        
        # Ensure embeddings are numpy arrays (in case they were loaded as strings from DB)
        if verbose:
            print("🔧 Converting embeddings to numpy arrays...")
        
        def ensure_numpy_array(emb):
            if isinstance(emb, str):
                # Convert string representation back to numpy array
                try:
                    # Try different parsing methods
                    if emb.startswith('[') and emb.endswith(']'):
                        # Handle array string like "[1.0 2.0 3.0]"
                        emb_clean = emb.strip('[]').replace('\n', ' ').replace('  ', ' ')
                        return np.fromstring(emb_clean, sep=' ')
                    else:
                        # Handle space-separated string like "1.0 2.0 3.0"
                        return np.fromstring(emb, sep=' ')
                except:
                    try:
                        # If fromstring fails, try eval (less safe but sometimes necessary)
                        return np.array(eval(emb))
                    except:
                        # Last resort: try literal_eval for safety
                        import ast
                        return np.array(ast.literal_eval(emb))
            elif isinstance(emb, (list, tuple)):
                return np.array(emb)
            elif isinstance(emb, np.ndarray):
                return emb
            else:
                # If it's already a proper numeric type, convert to array
                return np.array(emb)
        
        new_articles['embedding'] = new_articles['embedding'].apply(ensure_numpy_array)
        context_articles['embedding'] = context_articles['embedding'].apply(ensure_numpy_array)
        
        # Verify conversion worked
        if verbose:
            print(f"   • Sample new article embedding type: {type(new_articles.iloc[0]['embedding'])}")
            print(f"   • Sample new article embedding shape: {new_articles.iloc[0]['embedding'].shape if hasattr(new_articles.iloc[0]['embedding'], 'shape') else 'No shape'}")
            print(f"   • Sample context article embedding type: {type(context_articles.iloc[0]['embedding'])}")
            print(f"   • Sample context article embedding shape: {context_articles.iloc[0]['embedding'].shape if hasattr(context_articles.iloc[0]['embedding'], 'shape') else 'No shape'}")
        
        # 4. Convert clusters to simulate format (now that embeddings are available)
        (cluster_centers, cluster_emb_sum_dics, cluster_tf_sum_dics,
         cluster_topN_indices, cluster_topN_scores, cluster_topN_probs, cluster_id_mapping) = convert_clusters_to_simulate_format(
            existing_clusters, context_articles, window_size, verbose
        )
        
        original_cluster_count = len(cluster_centers)
        
        # 5. Build to_date for time-aware clustering
        to_date = pd.Timestamp(datetime.now().date())  # Use pandas Timestamp to match article dates
        
        # 5a. Assign to existing clusters
        if verbose:
            print("🎯 Assigning articles to existing clusters...")
            print(f"   • New articles to assign: {len(new_articles)}")
            print(f"   • Existing cluster centers: {len(cluster_centers)}")
            print(f"   • Article columns: {list(new_articles.columns)}")
            if len(new_articles) > 0:
                print(f"   • Sample article has embedding: {'embedding' in new_articles.columns}")
        
        # Convert cluster centers to numpy array if needed
        if len(cluster_centers) > 0 and not isinstance(cluster_centers, np.ndarray):
            cluster_centers = np.array(cluster_centers)
        
        window, cluster_emb_sum_dics, cluster_tf_sum_dics, _ = assign_to_clusters(
            initial=True,  # Use initial=True to consider all existing clusters
            verbose=verbose,
            window=new_articles,
            window_size=window_size,
            to_date=to_date,
            cluster_centers=cluster_centers,
            cluster_emb_sum_dics=cluster_emb_sum_dics,
            cluster_tf_sum_dics=cluster_tf_sum_dics,
            cluster_topN_probs=cluster_topN_probs,
            T=T,
            time_aware=time_aware,
            theme_aware=theme_aware,
            cluster_topN_indices=cluster_topN_indices,
            cluster_topN_scores=cluster_topN_scores
        )
        
        # 5b. Cluster any outliers (articles not assigned to existing clusters)
        outliers = window[window['cluster'] == -1]
        if len(outliers) >= min_articles:
            if verbose:
                print(f"🔍 Clustering {len(outliers)} outlier articles into new clusters...")
            
            window, cluster_centers, cluster_emb_sum_dics, cluster_tf_sum_dics, _ = cluster_outliers(
                window=window,
                cluster_centers=cluster_centers,
                cluster_emb_sum_dics=cluster_emb_sum_dics,
                cluster_tf_sum_dics=cluster_tf_sum_dics,
                min_articles=min_articles,
                verbose=verbose
            )
        elif len(outliers) > 0:
            if verbose:
                print(f"⚠️  {len(outliers)} outliers found but below threshold of {min_articles} - keeping as outliers")
        
        # 6. Save results to database
        if verbose:
            print("💾 Saving clustering results to database...")
        
        success = save_clustering_results(
            window, cluster_centers, original_cluster_count, verbose
        )
        
        if success:
            assigned_count = len(window[window['cluster'] >= 0])
            outlier_count = len(window[window['cluster'] == -1])
            if verbose:
                print("=" * 60)
                print("✅ Daily clustering completed successfully!")
                print(f"📊 Final statistics:")
                print(f"   • Articles processed: {len(window)}")
                print(f"   • Articles assigned: {assigned_count}")
                print(f"   • Outlier articles: {outlier_count}")
                print(f"   • Total clusters: {len(cluster_centers)}")
                print(f"   • Original clusters: {original_cluster_count}")
                print(f"   • New clusters created: {len(cluster_centers) - original_cluster_count}")
        else:
            print("❌ Failed to save clustering results")
            return False
        
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ Daily clustering failed: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 