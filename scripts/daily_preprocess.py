#!/usr/bin/env python3
"""
daily_preprocess.py

Daily preprocessing script for new articles.
Processes articles created today from Supabase.
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocess import main_daily

def main():
    """Process articles created today."""
    
    print("🚀 Starting daily preprocessing")
    print("=" * 60)
    print(f"📅 Processing articles from: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    try:
        # Process today's articles
        main_daily()
        
        print("\n✅ Daily preprocessing completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Daily preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 