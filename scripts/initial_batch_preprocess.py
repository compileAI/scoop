#!/usr/bin/env python3
"""
initial_batch_preprocess.py

Initial batch preprocessing script for articles from the last N days.
Processes articles from Supabase for cold-start.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocess import main_all, main_days

def main():
    """Process articles from the last N days for initial batch preprocessing."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Initial batch preprocessing for articles from the last N days')
    parser.add_argument('--days', type=int, default=30, 
                       help='Number of days ago to include articles from (default: 30)')
    parser.add_argument('--all', action='store_true',
                       help='Process all articles instead of last N days')
    
    args = parser.parse_args()
    
    if args.all:
        print("🚀 Starting initial batch preprocessing")
        print("=" * 60)
        print(f"📅 Processing all articles (cold start)")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        try:
            # Process all articles
            main_all()
            
            print("\n✅ Initial batch preprocessing completed successfully!")
            print("=" * 60)
            print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Initial batch preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("🚀 Starting initial batch preprocessing")
        print("=" * 60)
        print(f"📅 Processing articles from the last {args.days} days")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        try:
            # Process articles from last N days
            main_days(args.days)
            
            print("\n✅ Initial batch preprocessing completed successfully!")
            print("=" * 60)
            print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Initial batch preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 