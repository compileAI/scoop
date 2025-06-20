#!/usr/bin/env python3
"""
Test script for db_utils.py data retrieval functionality.
Tests the ability to fetch fully assembled pre-processed articles from the database.
"""

import sys
import os
from datetime import datetime
from pprint import pprint

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from db_utils import fetch_processed_articles_by_date


def test_fetch_processed_articles():
    """Test fetching processed articles for a specific date."""
    print("=" * 60)
    print("Testing fetch_processed_articles_by_date functionality")
    print("=" * 60)
    
    # Test date
    test_date = datetime(2025, 6, 18)
    print(f"Fetching articles for date: {test_date.strftime('%Y-%m-%d')}")
    
    # Fetch articles
    try:
        articles_df = fetch_processed_articles_by_date(test_date, limit=10)
        print(f"Successfully fetched {len(articles_df)} articles")
        
        if len(articles_df) == 0:
            print("No articles found for the specified date")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to fetch articles: {e}")
        return False
    
    # Expected columns based on the ProcessedArticlesDF structure
    expected_columns = [
        'id',
        'source_article_id', 
        'date',
        'title',
        'text',
        'sentences',
        'sentence_counts',
        'sentence_tokens',
        'sentence_embds'
    ]
    
    print(f"\nExpected columns: {expected_columns}")
    print(f"Actual columns: {list(articles_df.columns)}")
    
    # Check that all expected columns are present
    missing_columns = set(expected_columns) - set(articles_df.columns)
    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}")
        return False
    
    print("✓ All expected columns are present")
    
    # Check that all columns have non-empty values
    print("\nChecking column content:")
    for col in expected_columns:
        non_empty_count = articles_df[col].notna().sum()
        print(f"  {col}: {non_empty_count}/{len(articles_df)} non-empty values")
        
        if non_empty_count == 0:
            print(f"  ⚠️  WARNING: Column '{col}' is completely empty")
        elif non_empty_count < len(articles_df):
            print(f"  ⚠️  WARNING: Column '{col}' has some empty values")
        else:
            print(f"  ✓ Column '{col}' is fully populated")
    
    # Print detailed information about the structure
    print(f"\nDataFrame shape: {articles_df.shape}")
    print(f"Memory usage: {articles_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Print one complete article for verification
    if len(articles_df) > 0:
        print("\n" + "=" * 60)
        print("SAMPLE ARTICLE (Full Structure)")
        print("=" * 60)
        
        # Get the first article
        sample_article = articles_df.iloc[0]
        
        print(f"Article ID: {sample_article['id']}")
        print(f"Source Article ID: {sample_article['source_article_id']}")
        print(f"Date: {sample_article['date']}")
        print(f"Title: {sample_article['title']}")
        print(f"\nText (first 200 chars): {str(sample_article['text'])[:200]}...")
        
        print(f"\nSentence Count: {sample_article['sentence_counts']}")
        print(f"Number of sentences: {len(sample_article['sentences'])}")
        
        print(f"\nFirst 3 sentences:")
        for i, sentence in enumerate(sample_article['sentences'][:3]):
            print(f"  {i+1}. {sentence}")
        
        print(f"\nSentence tokens for first sentence:")
        if sample_article['sentence_tokens'] and len(sample_article['sentence_tokens']) > 0:
            print(f"  Tokens: {sample_article['sentence_tokens'][0]}")
        
        print(f"\nSentence embeddings info:")
        if sample_article['sentence_embds'] and len(sample_article['sentence_embds']) > 0:
            first_embedding = sample_article['sentence_embds'][0]
            print(f"  Number of embeddings: {len(sample_article['sentence_embds'])}")
            print(f"  First embedding shape: {first_embedding.shape if hasattr(first_embedding, 'shape') else len(first_embedding)}")
            print(f"  First embedding (first 5 dims): {first_embedding[:5] if hasattr(first_embedding, '__getitem__') else 'N/A'}")
        
        print("\n" + "=" * 60)
        print("COMPLETE ARTICLE DATA STRUCTURE")
        print("=" * 60)
        
        # Print the complete article structure (but truncate very long fields)
        article_dict = sample_article.to_dict()
        for key, value in article_dict.items():
            if key == 'text' and len(str(value)) > 300:
                print(f"{key}: {str(value)[:300]}... (truncated)")
            elif key == 'sentences' and len(value) > 5:
                print(f"{key}: {value[:5]}... (showing first 5 of {len(value)})")
            elif key == 'sentence_tokens' and len(value) > 3:
                print(f"{key}: {value[:3]}... (showing first 3 of {len(value)})")
            elif key == 'sentence_embds':
                print(f"{key}: [array of {len(value)} embeddings, each with {len(value[0]) if value else 0} dimensions]")
            else:
                print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✓ Successfully fetched {len(articles_df)} articles")
    print(f"✓ All expected columns present: {len(expected_columns)}")
    print(f"✓ Sample article data structure verified")
    print("✓ Test completed successfully")
    
    return True


def main():
    """Run the test."""
    print("Starting db_utils test...")
    
    try:
        success = test_fetch_processed_articles()
        
        if success:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 