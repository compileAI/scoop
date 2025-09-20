#!/usr/bin/env python3
"""
initial_batch_clustering.py

Simple script to run clustering.simulate() on articles from the last N days
and save the results to the database.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from clustering import simulate
from db_utils import save_clustering_results_to_db

def main():
    """Run clustering.simulate on articles from the last N days."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Initial batch clustering for articles from the last N days')
    parser.add_argument('--days', type=int, default=30, 
                       help='Number of days ago to include articles from (default: 30)')
    parser.add_argument('--window-size', type=int, default=14,
                       help='Window size in days (default: 14)')
    parser.add_argument('--slide-size', type=int, default=7,
                       help='Slide size in days (default: 7)')
    parser.add_argument('--num-windows', type=int, default=10,
                       help='Number of sliding windows (default: 10)')
    parser.add_argument('--min-articles', type=int, default=6,
                       help='Minimum articles per cluster (default: 6)')
    
    args = parser.parse_args()
    
    # Clustering parameters
    window_size = args.window_size
    slide_size = args.slide_size
    num_windows = args.num_windows
    min_articles = args.min_articles
    N = 10                  # top N keywords
    T = 4                   # similarity threshold parameter
    keyword_score = "tfidf" # keyword scoring method
    verbose = True          # verbose output
    time_aware = True       # time-aware clustering
    theme_aware = True      # theme-aware clustering
    
    # Calculate date range for last N days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Format dates as strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print("🚀 Starting clustering simulation")
    print("=" * 60)
    print(f"📅 Date range: {start_date_str} to {end_date_str} (last {args.days} days)")
    print(f"📊 Parameters:")
    print(f"   • Window size: {window_size} days")
    print(f"   • Slide size: {slide_size} days")
    print(f"   • Number of windows: {num_windows}")
    print(f"   • Min articles per cluster: {min_articles}")
    print(f"   • T parameter: {T}")
    print(f"   • Top N keywords: {N}")
    print(f"   • Keyword scoring: {keyword_score}")
    print(f"   • Time aware: {time_aware}")
    print(f"   • Theme aware: {theme_aware}")
    print("=" * 60)
    
    try:
        # Call clustering.simulate
        result = simulate(
            start_date=start_date_str,
            end_date=end_date_str,
            window_size=window_size,
            slide_size=slide_size,
            num_windows=num_windows,
            min_articles=min_articles,
            N=N,
            T=T,
            keyword_score=keyword_score,
            verbose=verbose,
            time_aware=time_aware,
            theme_aware=theme_aware
        )
        
        # Unpack results
        (all_window, cluster_keywords_df, final_num_cluster, avg_win_proc_time,
         nmi, ami, ri, ari, precision, recall, fscore) = result
        
        print("\n✅ Clustering simulation completed successfully!")
        print("=" * 60)
        print(f"📊 Results:")
        print(f"   • Total articles processed: {len(all_window)}")
        print(f"   • Final number of clusters: {final_num_cluster}")
        print(f"   • Average window processing time: {avg_win_proc_time}s")
        print(f"   • Clustered articles: {len(all_window[all_window['cluster'] >= 0])}")
        print(f"   • Outlier articles: {len(all_window[all_window['cluster'] == -1])}")
        
        if len(cluster_keywords_df) > 0:
            print(f"   • Cluster keywords DataFrame shape: {cluster_keywords_df.shape}")
        
        print("=" * 60)
        
        # Save results to database
        print("\n💾 Saving clustering results to database...")
        save_success = save_clustering_results_to_db(all_window, cluster_keywords_df)
        
        if save_success:
            print("🎉 Clustering and database save completed successfully!")
        else:
            print("⚠️  Clustering completed but database save failed!")
            
        return save_success
        
    except Exception as e:
        print(f"\n❌ Clustering simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 