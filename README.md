### SCOOP: Semantic Clustering Of Online Publications

## Features

- **Automatic API Key Rotation**: Handle Gemini API rate limits gracefully by rotating between multiple API keys
- **Semantic Clustering**: Cluster articles using embeddings and time-aware/theme-aware clustering
- **Daily Processing**: Process new articles incrementally
- **Database Integration**: Store processed articles in Supabase and embeddings in Pinecone

## Setup

### Environment Variables

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

#### Gemini API Key Rotation (Recommended)

To handle rate limits, you can configure multiple Gemini API keys. The system will automatically rotate between them:

```bash
GOOGLE_API_KEY_1=your_first_api_key
GOOGLE_API_KEY_2=your_second_api_key
GOOGLE_API_KEY_3=your_third_api_key
GOOGLE_API_KEY_4=your_fourth_api_key
GOOGLE_API_KEY_5=your_fifth_api_key
```

**Benefits:**
- Automatic rotation when a key hits rate limit
- 30-second cooldown period when all keys are rate limited
- Transparent error handling and logging
- No code changes needed - works with existing scripts

**Fallback:** If you only have one key, you can still use `GOOGLE_API_KEY` (but you'll be limited by rate limits).

### Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -r requirements.txt
```

## Usage

### Daily Processing

Process unprocessed articles from the last 3 days (most recent first):

```bash
python scripts/daily_preprocess.py
```

**How it works:**
- Checks for articles in `source_articles` that aren't in `cleaned_source_articles`
- Processes articles one at a time, saving progress incrementally
- Starts with most recent articles and works backward
- Automatically resumes from where it left off if interrupted
- Stops gracefully if daily quota is hit (without losing progress)

### Initial Batch Processing

Process articles from the last N days:

```bash
# Process last 30 days
python scripts/initial_batch_preprocess.py --days 30

# Process all articles (cold start)
python scripts/initial_batch_preprocess.py --all
```

### Clustering

Run clustering on processed articles:

```bash
python scripts/daily_clustering.py
```

## API Rate Limit Handling

The system automatically handles Gemini API rate limits through intelligent key rotation:

### Per-Minute Quotas (Recoverable)
1. **Normal Operation**: Uses the first available API key
2. **Rate Limit Hit**: Automatically rotates to the next available key
3. **All Keys Limited**: Waits for API-suggested delay + 10s buffer (or 70s default)
4. **Retry**: Continues processing after cooldown

### Daily Quotas (Non-Recoverable)
1. **Detection**: Automatically detects `PerDay` quota exhaustion
2. **Graceful Exit**: Stops processing immediately to avoid wasting time
3. **Progress Saved**: All successfully processed articles are already in the database
4. **Auto-Resume**: Next run automatically picks up where it left off

### Safeguards
- **Max 10 cooldown cycles**: Prevents infinite loops on persistent issues
- **Incremental saves**: Progress saved after each article
- **Smart retry delays**: Uses API-suggested delays with buffers

### Logging
All rate limit events are logged with emojis for easy monitoring:
- 🔑 API manager initialization
- 🔄 Key rotation
- 🚫 Rate limit detection
- 📊 Retry delay parsing
- ⏳ Cooldown waiting
- ❌ Daily quota exhaustion
- ✅ Successful processing
