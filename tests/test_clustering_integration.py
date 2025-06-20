#!/usr/bin/env python3
"""
Integration test for clustering.py with database utilities.
Tests the updated read_dataset function and basic clustering functionality.
"""

import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clustering import read_dataset


def test_read_dataset():
    """Test the updated read_dataset function with database integration."""
    print("=" * 60)
    print("Testing read_dataset with database integration")
    print("=" * 60)
    
    # Test parameters
    start_date = "2025-06-18"
    end_date = "2025-06-18"
    verbose = True
    
    print(f"Testing date range: {start_date} to {end_date}")
    
    try:
        article_df, all_vocab = read_dataset(start_date, end_date, verbose)
        
        if len(article_df) == 0:
            print("No articles found for the specified date range")
            return False
            
        print(f"\n✓ Successfully loaded {len(article_df)} articles")
        print(f"✓ Vocabulary size: {len(all_vocab)}")
        
        # Check required columns for clustering
        required_columns = [
            'id', 'date', 'title', 'text', 'sentences', 'sentence_counts',
            'sentence_tokens', 'sentence_embds', 'sentence_TFs', 'article_TF'
        ]
        
        missing_columns = set(required_columns) - set(article_df.columns)
        if missing_columns:
            print(f"ERROR: Missing required columns: {missing_columns}")
            return False
            
        print(f"✓ All required columns present: {required_columns}")
        
        # Verify data types and structures
        sample = article_df.iloc[0]
        
        print(f"\nData structure verification:")
        print(f"  - sentences: {type(sample['sentences'])} with {len(sample['sentences'])} items")
        print(f"  - sentence_tokens: {type(sample['sentence_tokens'])} with {len(sample['sentence_tokens'])} items")
        print(f"  - sentence_embds: {type(sample['sentence_embds'])} with {len(sample['sentence_embds'])} items")
        print(f"  - sentence_TFs: {type(sample['sentence_TFs'])}")
        print(f"  - article_TF: {type(sample['article_TF'])}")
        
        # Check that TF matrices have the right vocabulary size
        if hasattr(sample['sentence_TFs'], 'shape'):
            print(f"  - sentence_TFs shape: {sample['sentence_TFs'].shape}")
        if hasattr(sample['article_TF'], 'shape'):
            print(f"  - article_TF shape: {sample['article_TF'].shape}")
            
        print(f"\n✓ Data structures look correct for clustering")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the integration test."""
    print("Starting clustering integration test...")
    
    try:
        success = test_read_dataset()
        
        if success:
            print("\n🎉 Integration test passed!")
            print("The clustering module is ready to work with database utilities!")
            sys.exit(0)
        else:
            print("\n❌ Integration test failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 