# YouTube Content Compliance Analyzer

A comprehensive FastAPI-based service that analyzes YouTube content (videos and channels) against predefined compliance categories using advanced LLM processing with concurrent pipeline architecture.

## Features

- 🔍 **Channel Analysis**: Analyze entire YouTube channels with configurable video limits
- 🎥 **Individual Video Analysis**: Analyze single YouTube videos for compliance violations
- 📊 **Bulk Processing**: Process multiple channels concurrently with advanced queue-based pipeline
- 🎯 **Compliance Scoring**: Score content from 0 to 1 against multiple compliance categories
- 🔍 **Evidence Detection**: Identify specific instances and evidence of violations
- 📈 **Detailed Reporting**: Comprehensive reports with examples, scores, and summaries
- 🤖 **Multi-LLM Support**: Switch between local LLM (Mistral) and OpenAI
- ⚡ **High-Performance**: Concurrent processing with 10 transcript workers and 10 LLM workers
- 📁 **CSV Export**: Download analysis results as CSV files
- 🚀 **Real-time Monitoring**: Live progress tracking for bulk operations

## Project Structure

```
il-compliance/
├── run_app.py                          # Main entry point
├── src/                                # Source code directory
│   ├── __init__.py                     # Package initialization
│   ├── app.py                          # FastAPI application
│   ├── youtube_analyzer.py             # YouTube data collection
│   ├── llm_analyzer.py                 # LLM-based content analysis
│   └── creator_processor.py            # Bulk processing pipeline
├── data/                               # Data files
│   └── YouTube_Controversy_Categories.csv
├── requirements.txt                    # Python dependencies
├── env.example                         # Environment configuration template
├── .env                               # Your environment configuration
└── README.md                          # This file
```

## Setup

### Prerequisites

- Python 3.8+
- Local Mistral LLM server running on port 1234 (or OpenAI API access)
- YouTube API key (optional, for enhanced functionality)

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd il-compliance
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create environment file:
   ```bash
   cp env.example .env
   ```

5. Edit `.env` file with your settings:
   ```bash
   # LLM Settings
   LOCAL_LLM_URL=http://localhost:1234/v1
   OPENAI_API_KEY=your_openai_api_key_here

   # Toggle between "local" or "openai"
   LLM_PROVIDER=openai

   # YouTube API (optional, for enhanced functionality)
   YOUTUBE_API_KEY=your_youtube_api_key_here
   ```

## Usage

### Starting the API Server

Run the application using the simple entry point:

```bash
python run_app.py
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### API Endpoints

#### Core Analysis

**`POST /api/analyze-creator`** - Analyze a YouTube creator's recent videos
```json
{
  "creator_url": "https://www.youtube.com/@ChannelHandle",
  "video_limit": 10,
  "llm_provider": "openai"
}
```

**`POST /analyze/video`** - Analyze a single YouTube video
```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "llm_provider": "openai"
}
```

**`POST /analyze/channel`** - Analyze a YouTube channel (legacy)
```json
{
  "channel_url": "https://www.youtube.com/c/ChannelName",
  "video_limit": 5,
  "llm_provider": "openai"
}
```

**`POST /api/analyze-multiple`** - Analyze multiple URLs (videos or channels)
```json
{
  "urls": [
    "https://www.youtube.com/@Channel1",
    "https://www.youtube.com/watch?v=VIDEO_ID",
    "https://www.youtube.com/@Channel2"
  ],
  "llm_provider": "openai"
}
```

#### Bulk Processing

**`POST /api/bulk-analyze`** - Upload CSV file for bulk channel analysis
- Upload a CSV file with channel URLs
- Returns a `job_id` for tracking progress
- Processes channels concurrently with advanced pipeline

**`GET /api/bulk-analyze/{job_id}`** - Get bulk analysis status
```json
{
  "job_id": "uuid-string",
  "status": "processing|completed|failed",
  "total_urls": 50,
  "processed_urls": 25,
  "failed_urls": []
}
```

**`GET /api/bulk-analyze/{job_id}/results`** - Get detailed results
- Returns comprehensive analysis results
- Includes success/failure breakdown
- Provides detailed error categorization

**`GET /api/bulk-analyze/{job_id}/csv`** - Download results as CSV
- Downloads processed results in CSV format
- Includes all compliance scores and metadata

#### Utility

**`GET /categories`** - List all compliance categories with definitions

**`GET /`** - API health check

**`GET /api/debug/jobs`** - Debug endpoint to list all processing jobs

### Response Format

All analysis endpoints return data in this format:

```json
{
  "channel_id": "UCxxxxx",
  "channel_name": "Channel Name",
  "channel_handle": "@channelhandle",
  "video_analyses": [
    {
      "video_id": "video123",
      "video_title": "Video Title",
      "video_url": "https://youtube.com/watch?v=video123",
      "analysis": {
        "video_id": "video123",
        "results": {
          "Category Name": {
            "score": 0.75,
            "justification": "Reason for score",
            "evidence": ["Quote from transcript"]
          }
        }
      }
    }
  ],
  "summary": {
    "Category Name": {
      "max_score": 0.75,
      "average_score": 0.65,
      "videos_with_violations": 2,
      "total_videos": 5,
      "examples": [...]
    }
  }
}
```

## Compliance Categories

The system analyzes content against categories defined in `data/YouTube_Controversy_Categories.csv`:

- Controversy or Cancelled Creators
- Inflammatory mentions of politics, religion, and social issues
- Military conflict and weapons
- Obscenity and inappropriate content
- Content that could lead to death/injury
- Drug or alcohol related content
- Adult/sexual content
- And many more...

### Scoring System

Each category is scored from 0 to 1:
- **0**: No violation detected
- **0.25-0.5**: Minor or ambiguous instances
- **0.75-1**: Clear violations

## Advanced Features

### Concurrent Processing Pipeline

The bulk analysis system uses a sophisticated queue-based pipeline:

- **10 Transcript Workers**: Concurrent YouTube data extraction
- **10 LLM Workers**: Parallel AI content analysis  
- **5 Result Workers**: Efficient result aggregation
- **Real-time Monitoring**: Queue depths and progress tracking
- **Comprehensive Statistics**: P50/P99 timing metrics

### Error Handling

Detailed error categorization for failed analyses:
- `invalid_channel`: Invalid/private URLs
- `no_transcripts`: Channels without available transcripts
- `llm_processing_failed`: AI analysis failures
- `transcript_processing_error`: Technical YouTube API errors

### Performance Monitoring

- Real-time queue depth monitoring
- Timing percentiles (P1, P50, P99)
- Throughput metrics (URLs/second)
- Comprehensive completion statistics

## Development

### Running with Auto-reload

For development with automatic reloading:

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### Testing Individual Components

Test the YouTube analyzer:
```python
from src.youtube_analyzer import YouTubeAnalyzer
analyzer = YouTubeAnalyzer()
```

Test the LLM analyzer:
```python
from src.llm_analyzer import LLMAnalyzer
llm = LLMAnalyzer(provider="openai")
```

## Troubleshooting

### Common Issues

1. **"No .env file found"**: Copy `env.example` to `.env` and configure your API keys
2. **YouTube IP blocking**: Wait for temporary block to expire or use different network
3. **No transcripts available**: YouTube videos must have captions/transcripts enabled
4. **OpenAI rate limits**: Reduce concurrent workers or add delays between requests

### Environment Variables

Required variables in `.env`:
```bash
OPENAI_API_KEY=sk-...              # Required for OpenAI provider
LOCAL_LLM_URL=http://localhost:1234/v1  # Required for local provider
LLM_PROVIDER=openai                # Default provider
YOUTUBE_API_KEY=...                # Optional, enhances functionality
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Your License Here] 