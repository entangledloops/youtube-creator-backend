# YouTube Content Compliance Analyzer

A backend service that analyzes YouTube content (videos and channels) against a predefined set of compliance categories, using LLM for content assessment.

## Features

- Analyze YouTube channel videos for compliance violations
- Analyze individual YouTube videos
- Score content from 0 to 1 against multiple compliance categories
- Identify specific instances and evidence of violations
- Provide detailed reports with examples and timestamps
- Easily switch between local LLM (Mistral) and OpenAI
- Analyze transcripts directly from CSV files (no YouTube API needed)

## Setup

### Prerequisites

- Python 3.8+
- Local Mistral LLM server running on port 1234 (or OpenAI API access)
- YouTube API key (optional, for enhanced functionality)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd youtube-compliance-analyzer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create environment file:
   ```
   cp env.example .env
   ```

4. Edit `.env` file with your settings:
   ```
   # LLM Settings
   LOCAL_LLM_URL=http://localhost:1234/v1
   OPENAI_API_KEY=your_openai_api_key_here

   # Toggle between "local" or "openai"
   LLM_PROVIDER=local

   # YouTube API (optional)
   YOUTUBE_API_KEY=your_youtube_api_key_here
   ```

## Usage

### Starting the API Server

Run the FastAPI application:

```
python app.py
```

The API will be available at http://localhost:8000

You can access the API documentation at http://localhost:8000/docs

### Using the CLI Tool

Analyze a YouTube channel:

```
python analyze_channel.py channel https://www.youtube.com/c/ChannelName -l 5 -p local
```

Analyze a single video:

```
python analyze_channel.py video https://www.youtube.com/watch?v=VIDEO_ID -p local
```

Options:
- `-l`, `--limit`: Number of videos to analyze from a channel (default: 5)
- `-p`, `--provider`: LLM provider: "local" (Mistral) or "openai" (default: local)
- `-o`, `--output`: Save results to file instead of printing to console

### Analyzing Transcripts from CSV

If you already have transcripts or can't access them via the YouTube API, you can analyze them directly from a CSV file:

1. Create a sample CSV template:
   ```
   python analyze_transcript_csv.py create-sample sample_transcripts.csv
   ```

2. Fill in the CSV with your transcripts and run the analysis:
   ```
   python analyze_transcript_csv.py analyze your_transcripts.csv -p local
   ```

The CSV file should contain the following columns:
- `video_id`: YouTube video ID (or any unique identifier)
- `video_title`: Title of the video
- `transcript`: The full transcript text
- `video_url`: (Optional) URL of the video

Options:
- `-p`, `--provider`: LLM provider: "local" (Mistral) or "openai" (default: local)
- `-o`, `--output`: Save results to file instead of printing to console

## API Endpoints

### `POST /analyze/channel`

Analyzes a YouTube channel.

Request body:
```json
{
  "channel_url": "https://www.youtube.com/c/ChannelName",
  "video_limit": 5,
  "llm_provider": "local"
}
```

### `POST /analyze/video`

Analyzes a single YouTube video.

Request body:
```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "llm_provider": "local"
}
```

### `GET /categories`

Returns the list of compliance categories with definitions.

## Compliance Categories

The tool analyzes content against categories defined in `YouTube_Controversy_Categories.csv`, including:

- Controversy or Cancelled Creators
- Inflammatory mentions of politics, religion, and social issues
- Military conflict
- Obscenity
- And many more...

Each category is scored from 0 to 1, where:
- 0: No violation detected
- 0.25-0.5: Minor or ambiguous instances
- 0.75-1: Clear violations

## Project Structure

- `app.py` - FastAPI web application
- `youtube_analyzer.py` - YouTube data collection module
- `llm_analyzer.py` - LLM-based content analysis
- `analyze_channel.py` - CLI tool for YouTube channels/videos
- `analyze_transcript_csv.py` - CLI tool for analyzing transcripts from CSV files
- `YouTube_Controversy_Categories.csv` - Compliance categories definitions

## Extending

You can extend the tool by:

1. Adding new categories to the CSV file
2. Implementing additional LLM providers in `llm_analyzer.py`
3. Adding more detailed analysis functions
4. Building a frontend UI that consumes the API 