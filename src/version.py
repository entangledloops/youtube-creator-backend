"""
Version information for YouTube Content Compliance Analyzer
"""

__version__ = "1.2.10"
__version_info__ = (1, 2, 10)
__build_date__ = "2025-05-30"

# Version history
VERSION_HISTORY = {
    "1.2.10": {
        "date": "2025-05-30",
        "changes": [
            "CRITICAL BUG FIX: Fixed category name mismatch causing video scores to be ignored in final results",
            "Added fuzzy category matching to handle LLM returning 'Adult content' vs expected '\"Adult\" content'",
            "Fixed issue where 0.5 scores for adult content were processed but not included in CSV exports",
            "Enhanced debugging to identify exact vs fuzzy category matches in summary creation",
            "Added comprehensive category mismatch detection and logging for better troubleshooting",
            "Applied fuzzy matching to both summary creation and CSV export fallback logic for consistency",
            "Fixed misleading comments that still referenced old >0.2 filtering threshold",
            "Video analysis results with valid scores (e.g., 0.5) are no longer ignored due to category name differences",
            "Improved category name normalization to handle quotes, apostrophes, and case differences",
            "Enhanced logging to show expected vs found categories and potential matches for debugging"
        ]
    },
    "1.2.9": {
        "date": "2025-05-30",
        "changes": [
            "CRITICAL BUG FIX: Fixed cache hits being treated as new LLM requests causing 429 errors",
            "Cache hits now properly use their cached LLM analysis results instead of triggering new OpenAI calls",
            "Fixed disconnect where videos had valid cached scores but overall creator score was 0.0",
            "Enhanced cache hit fallback logic to include transcript field and proper video analysis structure",
            "Added comprehensive debugging to track cache hit processing and identify when cached results are used",
            "Fixed root cause where cache hits showed valid scores in frontend but were ignored in aggregation",
            "CRITICAL BUG INVESTIGATION: Added comprehensive debugging for scores being ignored in final results",
            "Added structure debugging to trace exact video analysis data format and identify malformed responses",
            "Enhanced aggregation debugging to track every step of score calculation and category matching",
            "Added per-video category breakdown to identify which videos have valid scores that get lost",
            "Implemented detailed category aggregation logging showing exact match vs fuzzy match attempts", 
            "Added final summary debugging to show what categories made it through aggregation",
            "Enhanced overall score calculation debugging to track all score sources and identify where max score comes from",
            "Added error detection for videos that return error responses instead of valid analysis",
            "Improved logging to distinguish between videos with no categories vs videos with zero scores",
            "Added per-category aggregation traces to identify where valid scores are being dropped",
            "Fixed misleading comments that referenced old >0.2 filtering threshold (now using MIN_VIOLATION_SCORE=0.0)",
            "Added fuzzy category matching to handle LLM returning slightly different category names",
            "Applied consistent fuzzy matching to both summary creation and CSV export logic",
            "CRITICAL FIX: Fixed variable name collision in create_channel_summaries where 'analysis_results' parameter was being overwritten",
            "This fix prevents the summary creation from crashing and allows all channels to be processed correctly",
            "Added traceback logging to exception handler for better error debugging"
        ]
    },
    "1.2.8": {
        "date": "2025-05-30",
        "changes": [
            "MAJOR IMPROVEMENT: Centralized output directory configuration using get_output_directory() function",
            "Added OUTPUT_DIR environment variable support (defaults to 'output') for configurable file storage",
            "SEMANTIC CONSISTENCY: Updated all 'results' references to 'output' for consistent terminology",
            "Enhanced CSV upload processing with intelligent YouTube URL column detection",
            "Added automatic header detection for CSV files (handles files with or without headers)",
            "Multi-column CSV support - automatically selects column with most YouTube URLs",
            "Improved CSV error handling with detailed logging of detection process",
            "Added is_youtube_url() function with comprehensive YouTube URL pattern matching",
            "Fixed duplicate results_dir configuration by consolidating into single environment-driven function",
            "Updated .gitignore and env.example to reflect new OUTPUT_DIR configuration",
            "Improved user experience with better CSV upload error messages and guidance"
        ]
    },
    "1.2.7": {
        "date": "2025-05-29",
        "changes": [
            "PROPER FIX: Eliminated race condition in URL queuing without using sleep() calls",
            "Completion monitor now waits for all URLs to be queued before checking for missing URLs",
            "Fixed root cause where URLs were marked as 'missing' while still being added to the pipeline",
            "Removed band-aid sleep() calls in favor of proper synchronization using url_task",
            "URLs can no longer be incorrectly marked as 'pipeline tracking error' during startup",
            "Ensures deterministic behavior - all URLs are in the pipeline before completion checking begins"
        ]
    },
    "1.2.6": {
        "date": "2025-05-29",
        "changes": [
            "CRITICAL FIX: Fixed race condition where channels were counted twice (in both results and failed_urls)",
            "Fixed 'pipeline tracking error' for channels that were still being processed when completion check ran",
            "Added safeguard to only check for missing URLs when NO channels are still in processing stages",
            "Result worker now removes URLs from failed_urls if they complete successfully after being marked as missing",
            "Prevents double-counting that caused '10 out of 8 URLs processed' type errors",
            "Fixed timing issue where @handle URLs were marked as failed while still in controversy screening"
        ]
    },
    "1.2.5": {
        "date": "2025-05-29",
        "changes": [
            "CRITICAL FIX: Fixed double-counting URLs bug that caused 10/8 completion (125%)",
            "Added duplicate URL checks in job completion monitor to prevent adding URLs to failed_urls twice",
            "CRITICAL FIX: Fixed queue management bug where current_job_id wasn't properly shared between modules",
            "Changed current_job_id import to access through module reference instead of direct import",
            "Fixed Python module-level variable import issue that caused new jobs to queue indefinitely",
            "Enhanced debugging to track when URLs are added to failed_urls vs missing URL detection",
            "Prevents 'pipeline tracking error' duplicates by checking if URL already exists in failed_urls"
        ]
    },
    "1.2.4": {
        "date": "2025-05-29",
        "changes": [
            "Added comprehensive debugging for queue management and pipeline tracking issues",
            "Enhanced logging to track when URLs are added to failed_urls vs missing URL detection",
            "Added debugging for current_job_id state and job queue transitions",
            "Improved pipeline tracking error diagnosis with detailed failed_urls logging",
            "Added timing analysis for channel discovery failures vs job completion monitoring",
            "Enhanced debugging output for @handle URL resolution and channel access issues"
        ]
    },
    "1.2.3": {
        "date": "2025-05-29",
        "changes": [
            "CRITICAL FIX: Fixed queue management bug where current_job_id was reset to None but never set to next job",
            "start_next_job() now properly sets current_job_id to the next job ID instead of leaving it None",
            "Fixed issue where second job would queue even when no jobs were actually running",
            "Improved queue state management to properly track when jobs are running vs when queue is empty",
            "Added logging to indicate when queue is empty and ready for new jobs",
            "Removed redundant current_job_id reset logic that was causing the queue management bug"
        ]
    },
    "1.2.2": {
        "date": "2025-05-29",
        "changes": [
            "CRITICAL FIX: Fixed queue management bug where current_job_id wasn't reset after job completion",
            "New jobs no longer incorrectly wait in queue when no jobs are actually running",
            "Added DELETE /api/bulk-analyze/cancel-all endpoint to cancel all ongoing jobs and clear queue",
            "Cancel-all API provides detailed summary of cancelled jobs (active vs queued)",
            "Improved job completion cleanup to properly reset global job state",
            "Enhanced logging for job queue state transitions and cancellations"
        ]
    },
    "1.2.1": {
        "date": "2025-05-29",
        "changes": [
            "CRITICAL FIX: Fixed race condition in job completion logic that caused channels to be marked as failed",
            "Job completion now properly waits for all channels to finish processing, not just empty queues",
            "Fixed issue where cache hits completed instantly while normal processing was still ongoing",
            "Enhanced job completion monitor to check pipeline stages for channels still being processed",
            "Improved debugging output for job completion detection and pipeline tracking issues",
            "Prevents premature job completion when some channels are still in controversy screening or video processing",
            "Fixed 'pipeline tracking error' failures for channels that were actually still being processed"
        ]
    },
    "1.2.0": {
        "date": "2025-05-29",
        "changes": [
            "CRITICAL FIX: DISABLE_VIDEO_CACHE now properly disables all cache usage (was still checking cache)",
            "Added CACHE_TRANSCRIPTS_ONLY environment variable for transcript-only caching mode",
            "Transcript-only mode reuses cached transcripts but forces fresh LLM analysis",
            "Enhanced cache system with separate methods for full cache vs transcript-only cache",
            "Improved cache statistics to show current cache mode (disabled/transcript_only/full_cache)",
            "Fixed cache storage logic to respect environment flags properly",
            "Added proper logging to indicate cache mode and storage behavior",
            "Cache system now supports three modes: disabled, transcript-only, and full caching"
        ]
    },
    "1.1.9": {
        "date": "2025-05-29",
        "changes": [
            "CRITICAL FIX: Added channel name scraping fallback when YouTube API is blocked (403 errors)",
            "Enhanced controversy screening to extract creator names from URLs when API fails",
            "Fixed 'Unknown' channel names in controversy screening by scraping channel pages",
            "Added YouTube URL context to controversy screening for better LLM analysis",
            "Improved channel info extraction with multiple fallback methods for name resolution",
            "Enhanced controversy screening to handle cases where API returns no channel data",
            "Fixed controversy screening effectiveness when YouTube API access is restricted",
            "Added robust channel name extraction from og:title, page titles, and JSON-LD metadata"
        ]
    },
    "1.1.8": {
        "date": "2025-05-29",
        "changes": [
            "MAJOR FIX: Controversial channels now properly show as FAIL with High Risk (1.0) score instead of ERROR",
            "Controversial channels now display actual LLM controversy reasons instead of generic messages",
            "Added proper 'Controversy or Cancelled Creators' category scoring for controversial channels",
            "Controversial channels now appear in results section with proper analysis structure, not failed_urls",
            "Fixed CSV exports to show controversial channels as FAIL with 1.0 score and controversy category marked",
            "Enhanced controversy screening to create proper video analysis and summary structures",
            "Controversial channels now stay in results throughout the pipeline instead of being moved to failed_urls",
            "Improved job completion logic to handle controversial channels in results properly"
        ]
    },
    "1.1.7": {
        "date": "2025-05-29",
        "changes": [
            "MAJOR IMPROVEMENT: Fixed controversy handling to use actual LLM reasons instead of generic messages",
            "Controversial channels now immediately fail with specific controversy reasons from LLM analysis",
            "Changed controversy failures from 'ERROR' type to 'FAIL' type with proper categorization",
            "Enhanced missing URL detection to properly categorize different failure types",
            "Controversial channels no longer have their videos processed - they fail immediately with LLM reason",
            "Improved error messages to show actual controversy screening results",
            "Better distinction between controversy failures, processing failures, and screening errors"
        ]
    },
    "1.1.6": {
        "date": "2025-05-28",
        "changes": [
            "PROPER FIX: Fixed job completion tracking by ensuring all processed URLs are counted",
            "Removed forced completion timer in favor of proper channel completion tracking",
            "Added missing URL detection and automatic addition to failed_urls when queues are empty",
            "Fixed edge case where channels passed controversy screening but never got counted as completed",
            "Enhanced completion logic to handle channels with no successful video analyses",
            "Improved debugging output for completion detection issues",
            "Ensures frontend gets accurate progress updates by properly tracking all channel outcomes"
        ]
    },
    "1.1.5": {
        "date": "2025-05-28",
        "changes": [
            "CRITICAL FIX: Added forced completion mechanism for jobs stuck at 7/8 channels with empty queues",
            "Added detailed debugging for job completion detection to identify missing URLs",
            "Implemented 30-second timeout fallback to force completion when queues are empty but URLs missing",
            "Enhanced job completion monitor with missing URL detection and controversy check failure analysis",
            "Fixed edge case where channels with processed videos weren't being counted as completed",
            "Added comprehensive logging to debug completion detection issues",
            "Prevents indefinite hanging when pipeline stages don't properly track channel completion"
        ]
    },
    "1.1.4": {
        "date": "2025-05-28",
        "changes": [
            "CRITICAL FIX: Fixed cache hit processing that was causing jobs to hang at 7/8 completion",
            "Fixed 'Could not find result queue for cached video' error that prevented cache hits from being processed",
            "Improved cache hit handling by passing result_queue directly to video transcript workers",
            "Enhanced fallback logic for cache hits when queue lookup fails",
            "Fixed pipeline stage tracking for cache hits to properly flow through result processing",
            "Cache hits now properly increment videos_completed counter and trigger job completion",
            "Eliminated frontend hanging when cache hits occur by ensuring proper result queue processing"
        ]
    },
    "1.1.3": {
        "date": "2025-05-28",
        "changes": [
            "Increased transcript workers from 2 to 3 for improved processing speed",
            "Added DISABLE_VIDEO_CACHE environment variable to bypass caching during testing",
            "Fixed channel access issues for @handle URLs by implementing handle resolution",
            "Added scraping-based resolution for @handles, /c/ custom URLs, and /user/ legacy URLs",
            "Improved channel name extraction for all URL formats",
            "Enhanced error handling and logging for channel resolution failures",
            "Fixed 'Failed to access YouTube channel' errors for valid channels with handles"
        ]
    },
    "1.1.2": {
        "date": "2025-05-28",
        "changes": [
            "Fixed critical CSV export scoring bug - scores now properly calculated from video analyses when summary is missing",
            "Added fallback score calculation for cancelled/partial jobs that don't have summary data",
            "CSV exports now show correct scores instead of all zeros for incomplete jobs",
            "Enhanced CSV export logging to indicate when fallback calculation is used",
            "Improved score calculation accuracy for partial results downloads"
        ]
    },
    "1.1.1": {
        "date": "2025-05-28",
        "changes": [
            "Fixed transcript queue stuck issue when retries are exhausted - properly handle task_done() calls",
            "Fixed cache hit frontend update issue - improved result queue lookup and fallback handling",
            "Fixed YouTube API counter accuracy - only count actual successful API requests, not hypothetical calls",
            "Improved error handling in transcript worker retry logic to prevent deadlocks",
            "Enhanced cache hit processing to properly flow through pipeline stages",
            "Removed misleading API call tracking for unimplemented handle/username resolution",
            "Added better logging and error detection for YouTube API call failures"
        ]
    },
    "1.1.0": {
        "date": "2025-05-28",
        "changes": [
            "Added SQLite-based video cache system to store transcripts and LLM results for 30-day reuse",
            "Implemented automatic cache lookup before transcript fetching to skip duplicate processing",
            "Added cache hit tracking and reporting in job progress and detailed logging",
            "Created cache statistics endpoint (/api/cache/stats) for monitoring cache performance",
            "Enhanced YouTube API call logging and error handling for better debugging",
            "Fixed YouTube API counter tracking with improved logging and error detection",
            "Added automatic cache expiration and cleanup of entries older than 30 days",
            "Cached results bypass transcript and LLM processing, going directly to results queue",
            "Cache system uses URL hashing for fast lookups and includes database indexing",
            "Added cache hit counter to video progress tracking in job status responses"
        ]
    },
    "1.0.4": {
        "date": "2025-05-28",
        "changes": [
            "Fixed task_done() error in job cancellation by tracking items removed from queues",
            "Fixed 'completed' counter issue by properly moving items from result_processing to completed stage",
            "Improved transcript worker retry logic to handle YouTube rate limiting without permanent failures",
            "Added better error handling for transcript failures that aren't rate limit related",
            "Reduced verbose logging in status endpoint and transcript failures to minimize log noise",
            "Increased YouTube rate limiting backoff times (180s initial, 900s max) to prevent IP bans",
            "Reduced max concurrent transcript requests from 5 to 2 and increased delay from 2.0s to 3.0s",
            "Added deadlock detection to pipeline monitor for stuck transcript queues",
            "Added timeout handling to video transcript workers to prevent indefinite blocking",
            "Updated environment variable documentation for new rate limiting settings"
        ]
    },
    "1.0.3": {
        "date": "2025-05-27",
        "changes": [
            "Fixed YouTube API call tracking - now properly counts all API calls made",
            "Fixed channel name extraction for channel ID URLs to ensure controversy screening works",
            "Added detailed debug logging for controversy screening to track LLM responses",
            "Improved channel info extraction to always get channel name even from direct channel ID URLs",
            "Fixed import issue for API call tracking in youtube_analyzer.py",
            "Refactored app.py into logical modules: pipeline_workers, job_manager, controversy_screener, export_handlers",
            "Moved rate limiter to separate module to avoid circular imports"
        ]
    },
    "1.0.2": {
        "date": "2025-05-27",
        "changes": [
            "Added separate controversy screening queue in the pipeline for better tracking and error handling",
            "Fixed controversy screening to properly distinguish between flagged creators (fail) vs screening errors (continue)",
            "Added controversy_check_failures tracking to identify channels processed despite screening errors",
            "Improved pipeline stage tracking with new stages: queued_for_controversy, screening_controversy, controversy_check_failed",
            "Enhanced single creator analysis to include controversy screening before processing videos",
            "Fixed worker cleanup to properly await cancelled tasks and prevent 'Task was destroyed' warnings",
            "Fixed variable name collision between result_worker function and result_worker_task variable",
            "Made controversy screening more specific to flag only current, ongoing controversies",
            "Added debug logging throughout the pipeline for better troubleshooting"
        ]
    },
    "1.0.1": {
        "date": "2025-05-25",
        "changes": [
            "Fixed duplicate task_done() calls in channel discovery worker",
            "Added proper error handling for invalid channel URLs",
            "Improved queue management to prevent synchronization issues",
            "Added debug logging for queue operations",
            "Fixed 'task_done() called too many times' error"
        ]
    },
    "1.0.0": {
        "date": "2025-05-23",
        "initial_release": True,
        "features": [
            "YouTube channel and video analysis",
            "Content compliance checking across multiple categories",
            "Bulk analysis with CSV upload",
            "Real-time progress tracking",
            "Export results as CSV or JSON evidence",
            "Support for OpenAI and local LLM providers",
            "Queue-based processing pipeline",
            "Controversy pre-screening for creators"
        ]
    }
} 