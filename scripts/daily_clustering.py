#!/usr/bin/env python3
"""
daily_clustering.py

Daily incremental clustering script.
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
from db_utils import (
    fetch_processed_articles_since_date,
    load_active_clusters,
    save_cluster,
    assign_articles_to_cluster
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'daily_clustering_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("daily_clustering")

class DailyClusteringProcessor:
    """Handles daily incremental clustering of new articles."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        
    def get_new_articles_since_date(self, since_date: datetime, limit: int = None) -> pd.DataFrame:
        """Get articles that were processed since a specific date."""
        log.info(f"📖 Loading articles since {since_date.strftime('%Y-%m-%d')}...")
        
        try:
            df = fetch_processed_articles_since_date(since_date, limit=limit)
            
            if len(df) == 0:
                log.info("📭 No new articles found")
                return pd.DataFrame()
            
            log.info(f"✅ Found {len(df)} new articles")
            return df
            
        except Exception as e:
            log.error(f"❌ Error loading new articles: {e}")
            return pd.DataFrame()
    
    def run_daily_clustering(self, days_back: int = 1, limit: int = None, dry_run: bool = False) -> bool:
        """Run daily clustering."""
        
        log.info("🚀 Starting daily clustering process")
        log.info("=" * 60)
        
        try:
            since_date = datetime.now() - timedelta(days=days_back)
            log.info(f"📅 Processing articles since: {since_date.strftime('%Y-%m-%d %H:%M:%S')}")
            
            new_articles = self.get_new_articles_since_date(since_date, limit=limit)
            
            if len(new_articles) == 0:
                log.info("📭 No new articles to process")
                return True
            
            active_clusters = load_active_clusters()
            
            if dry_run:
                log.info("🧪 Dry run mode - would process:")
                log.info(f"   📰 {len(new_articles)} new articles")
                log.info(f"   🎯 {len(active_clusters)} active clusters")
                return True
            
            log.info("🏁 Daily clustering setup completed!")
            
            return True
            
        except Exception as e:
            log.error(f"❌ Daily clustering failed: {e}")
            return False

def main():
    """Main entry point for daily clustering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily incremental clustering")
    parser.add_argument("--days-back", type=int, default=1, help="Process articles from N days back")
    parser.add_argument("--limit", type=int, help="Limit number of articles (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Run without making changes")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    
    args = parser.parse_args()
    
    processor = DailyClusteringProcessor(similarity_threshold=args.threshold)
    
    success = processor.run_daily_clustering(
        days_back=args.days_back,
        limit=args.limit,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 