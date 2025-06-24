#!/usr/bin/env python3
"""
initial_batch_preprocess.py

Initial batch preprocessing script for all articles.
Processes all articles from Supabase for cold-start.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocess import main_all

def main():
    """Process all articles from Supabase for initial batch preprocessing."""
    
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

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 