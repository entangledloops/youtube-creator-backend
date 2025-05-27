"""
Job queue management and processing for bulk analysis
"""
import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import os
import traceback

from src.youtube_analyzer import YouTubeAnalyzer
from src.llm_analyzer import LLMAnalyzer
from src.pipeline_workers import (
    channel_discovery_worker, controversy_screening_worker,
    video_transcript_worker, llm_worker, result_worker,
    monitor_pipeline_detailed, update_pipeline_stage,
    clear_queues
)

logger = logging.getLogger(__name__)

# Global job queue for sequential processing
job_queue = []
current_job_id = None
job_queue_lock = asyncio.Lock()

# Global task tracking for cancellation
active_job_tasks = {}  # job_id -> list of asyncio tasks

async def process_creators_pipeline(job_id: str, urls: List[str], video_limit: int, llm_provider: str, analysis_results: dict):
    """Queue-based pipeline with proper video-level rate limiting - NON-BLOCKING"""
    try:
        logger.info(f"🚀 Starting pipeline for job {job_id} with {len(urls)} URLs")
        
        # Update job status
        analysis_results[job_id]['status'] = 'processing'
        analysis_results[job_id]['started_at'] = datetime.now().isoformat()
        
        # Create queues
        channel_queue = asyncio.Queue()
        controversy_queue = asyncio.Queue()
        video_queue = asyncio.Queue()
        llm_queue = asyncio.Queue()
        result_queue = asyncio.Queue()
        
        # Store queues for status updates
        queues = {
            'channel_queue': channel_queue,
            'controversy_queue': controversy_queue,
            'video_queue': video_queue,
            'transcript_queue': video_queue,  # Alias for status endpoint
            'llm_queue': llm_queue,
            'result_queue': result_queue
        }
        
        # Initialize analyzers
        youtube_analyzer = YouTubeAnalyzer()
        llm_analyzer = LLMAnalyzer(provider=llm_provider)
        
        # Initialize timing stats
        timing_stats = {
            'channel_discovery': [],
            'controversy_screening': [],
            'transcript_fetch': [],
            'llm_analysis': []
        }
        
        # Create worker tasks
        workers = []
        
        # Channel discovery workers
        for i in range(3):
            worker = asyncio.create_task(
                channel_discovery_worker(i, channel_queue, controversy_queue, youtube_analyzer, timing_stats, job_id, analysis_results)
            )
            # Create closure to properly capture queues
            def make_get_queues(q):
                return lambda: q
            def make_clear_queues(q):
                return lambda: clear_queues(q)
            worker.get_queues = make_get_queues(queues)
            worker.clear_queues = make_clear_queues(queues)
            workers.append(worker)
        
        # Controversy screening workers
        for i in range(2):
            worker = asyncio.create_task(
                controversy_screening_worker(i, controversy_queue, video_queue, llm_analyzer, timing_stats, job_id, analysis_results)
            )
            worker.get_queues = make_get_queues(queues)
            worker.clear_queues = make_clear_queues(queues)
            workers.append(worker)
        
        # Video transcript workers
        for i in range(5):
            worker = asyncio.create_task(
                video_transcript_worker(i, video_queue, llm_queue, youtube_analyzer, timing_stats, job_id, analysis_results)
            )
            worker.get_queues = make_get_queues(queues)
            worker.clear_queues = make_clear_queues(queues)
            workers.append(worker)
        
        # LLM workers
        for i in range(2):
            worker = asyncio.create_task(
                llm_worker(i, llm_queue, result_queue, llm_analyzer, timing_stats, job_id, analysis_results)
            )
            worker.get_queues = make_get_queues(queues)
            worker.clear_queues = make_clear_queues(queues)
            workers.append(worker)
        
        # Result worker
        result_worker_task = asyncio.create_task(
            result_worker(0, result_queue, timing_stats, job_id, analysis_results)
        )
        result_worker_task.get_queues = make_get_queues(queues)
        result_worker_task.clear_queues = make_clear_queues(queues)
        workers.append(result_worker_task)
        
        # Monitor task
        monitor_task = asyncio.create_task(
            monitor_pipeline_detailed(
                job_id, channel_queue, controversy_queue, video_queue, 
                llm_queue, result_queue, len(urls), timing_stats, analysis_results
            )
        )
        monitor_task.get_queues = make_get_queues(queues)
        monitor_task.clear_queues = make_clear_queues(queues)
        workers.append(monitor_task)
        
        # Completion monitor task - this will handle job completion
        completion_task = asyncio.create_task(
            monitor_job_completion(job_id, urls, workers, queues, analysis_results)
        )
        workers.append(completion_task)
        
        # Store all workers for cancellation
        active_job_tasks[job_id] = workers
        
        # Create a task to add URLs to the queue asynchronously
        async def add_urls_to_queue():
            """Add URLs to channel queue asynchronously"""
            try:
                logger.info(f"📊 Starting to queue {len(urls)} channels for processing")
                for i, url in enumerate(urls):
                    # Update pipeline stage to show URLs being queued
                    update_pipeline_stage(job_id, None, 'queued_for_discovery', analysis_results=analysis_results)
                    
                    await channel_queue.put({
                        'url': url,
                        'video_limit': video_limit,
                        'start_time': time.time()
                    })
                    
                    # Log progress every 10 URLs
                    if (i + 1) % 10 == 0:
                        logger.debug(f"📊 Queued {i + 1}/{len(urls)} URLs for discovery")
                        
                logger.info(f"✅ Finished queuing all {len(urls)} URLs for job {job_id}")
            except Exception as e:
                logger.error(f"Error adding URLs to queue: {str(e)}")
        
        # Start the URL queuing task - NON-BLOCKING
        url_task = asyncio.create_task(add_urls_to_queue())
        workers.append(url_task)  # Add to workers list so it gets cancelled if job is cancelled
        
        # DO NOT WAIT FOR WORKERS - Let them run in background
        logger.info(f"✅ Pipeline started for job {job_id} - workers running in background")
        
    except Exception as e:
        logger.error(f"Error starting pipeline for job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Mark job as failed
        analysis_results[job_id]['status'] = 'failed'
        analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
        analysis_results[job_id]['error'] = str(e)
        
        # Clean up
        if job_id in active_job_tasks:
            del active_job_tasks[job_id]

async def monitor_job_completion(job_id: str, urls: List[str], workers: List[asyncio.Task], 
                               queues: dict, analysis_results: dict):
    """Monitor job completion without blocking"""
    try:
        while True:
            await asyncio.sleep(2)  # Check every 2 seconds
            
            # Check if job is cancelled
            if analysis_results[job_id]['status'] in ['cancelled', 'cancelling']:
                logger.info(f"🛑 Job {job_id} cancelled, stopping completion monitor")
                break
            
            # Check if all work is done
            all_queues_empty = all(q.empty() for q in queues.values())
            
            # Check completion based on processed URLs
            total_processed = len(analysis_results[job_id]['results']) + len(analysis_results[job_id]['failed_urls'])
            all_urls_processed = total_processed >= len(urls)
            
            if all_queues_empty and all_urls_processed:
                logger.info(f"✅ Job {job_id} completed - all URLs processed")
                
                # Mark job as completed
                analysis_results[job_id]['status'] = 'completed'
                analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
                analysis_results[job_id]['processed_urls'] = total_processed
                
                # Create channel summaries
                await create_channel_summaries(job_id, analysis_results)
                
                # Cancel all workers
                for worker in workers:
                    if not worker.done():
                        worker.cancel()
                
                # Clean up
                if job_id in active_job_tasks:
                    del active_job_tasks[job_id]
                
                # Start next job in queue
                await start_next_job(analysis_results)
                break
                
    except Exception as e:
        logger.error(f"Error in completion monitor for job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())

async def start_next_job(analysis_results: dict):
    """Start the next job in queue if any"""
    async with job_queue_lock:
        global current_job_id
        current_job_id = None
        
        if job_queue:
            next_job = job_queue.pop(0)
            current_job_id = next_job['job_id']
            
            # Update status
            analysis_results[current_job_id]['status'] = 'processing'
            analysis_results[current_job_id]['queue_position'] = 0
            
            logger.info(f"🚀 Starting next queued job {current_job_id}")
            
            # Start processing - NON-BLOCKING
            asyncio.create_task(
                process_creators_pipeline(
                    current_job_id,
                    next_job['urls'],
                    next_job['video_limit'],
                    next_job['llm_provider'],
                    analysis_results
                )
            )

async def create_channel_summaries(job_id: str, analysis_results: dict):
    """Create channel summaries from individual video results"""
    try:
        logger.info(f"📊 Creating channel summaries for job {job_id}")
        
        # Get categories for summary
        src_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(src_dir)
        categories_file = os.path.join(project_root, "data", "YouTube_Controversy_Categories.csv")
        categories_df = pd.read_csv(categories_file)
        all_categories = categories_df['Category'].tolist()
        
        # First, check for controversy failures that didn't make it to results
        controversy_failures = analysis_results[job_id].get('controversy_check_failures', {})
        for url, controversy_info in controversy_failures.items():
            # Check if this URL made it to results
            if url not in analysis_results[job_id]['results']:
                # This controversial channel didn't make it to results - add to failed URLs
                logger.warning(f"⚠️ Controversial channel {url} has no results - adding to failed URLs")
                
                # Check if it's already in failed_urls to avoid duplicates
                already_failed = any(f['url'] == url for f in analysis_results[job_id]['failed_urls'])
                if not already_failed:
                    analysis_results[job_id]['failed_urls'].append({
                        'url': url,
                        'error': f"Channel flagged for controversy: {controversy_info.get('reason', 'Unknown reason')}. No videos could be analyzed.",
                        'error_type': 'controversy_flagged_no_results',
                        'channel_name': controversy_info.get('channel_name', 'Unknown'),
                        'controversy_reason': controversy_info.get('reason', 'Unknown reason')
                    })
        
        # Process each channel's video results
        channels_to_remove = []
        
        for url, channel_data in analysis_results[job_id]['results'].items():
            # Check if this channel has any successfully analyzed videos
            video_analyses = channel_data.get('video_analyses', [])
            
            if not video_analyses:
                # No videos were successfully analyzed - this is a failure
                logger.warning(f"❌ Channel {url} has no successfully analyzed videos - moving to failed")
                
                # Check if this was a controversial channel
                is_controversial = channel_data.get('controversy_flagged', False)
                controversy_reason = channel_data.get('controversy_reason', None)
                
                if is_controversial:
                    error_msg = f"Channel flagged for controversy: {controversy_reason}. Additionally, no videos could be analyzed."
                    error_type = 'controversy_flagged_no_videos'
                else:
                    error_msg = 'Failed to analyze any videos from this channel. All transcript downloads may have failed.'
                    error_type = 'no_videos_analyzed'
                
                # Add to failed URLs
                analysis_results[job_id]['failed_urls'].append({
                    'url': url,
                    'error': error_msg,
                    'error_type': error_type,
                    'channel_name': channel_data.get('channel_name', 'Unknown'),
                    'channel_id': channel_data.get('channel_id', 'unknown'),
                    'video_count': len(channel_data.get('original_videos', [])),
                    'controversy_flagged': is_controversial,
                    'controversy_reason': controversy_reason
                })
                
                # Mark for removal from results
                channels_to_remove.append(url)
                continue
                
            # Create summary across all videos for this channel
            summary = {}
            for category in all_categories:
                category_violations = []
                max_score = 0
                total_score = 0
                videos_with_violations = 0
                
                for video_analysis in video_analyses:
                    if category in video_analysis.get("analysis", {}).get("results", {}):
                        violation = video_analysis["analysis"]["results"][category]
                        score = violation.get("score", 0)
                        
                        if score > 0:
                            videos_with_violations += 1
                            max_score = max(max_score, score)
                            total_score += score
                            
                            category_violations.append({
                                "video_id": video_analysis["video_id"],
                                "video_title": video_analysis["video_title"],
                                "video_url": video_analysis["video_url"],
                                "score": score,
                                "evidence": violation.get("evidence", [])[0] if violation.get("evidence") else ""
                            })
                
                if videos_with_violations > 0:
                    summary[category] = {
                        "max_score": max_score,
                        "average_score": total_score / videos_with_violations,
                        "videos_with_violations": videos_with_violations,
                        "total_videos": len(video_analyses),
                        "examples": sorted(category_violations, key=lambda x: x["score"], reverse=True)[:5]
                    }
            
            # Update the channel data with summary
            analysis_results[job_id]['results'][url]['summary'] = summary
        
        # Remove failed channels from results
        for url in channels_to_remove:
            del analysis_results[job_id]['results'][url]
            
        # Update final processed count (count channels, not videos)
        analysis_results[job_id]['processed_urls'] = len(analysis_results[job_id]['results']) + len(analysis_results[job_id]['failed_urls'])
        
        logger.info(f"✅ Created summaries for {len(analysis_results[job_id]['results'])} channels")
        if channels_to_remove:
            logger.info(f"❌ Moved {len(channels_to_remove)} channels to failed (no videos analyzed)")
        
    except Exception as e:
        logger.error(f"💥 Error creating channel summaries: {str(e)}")

async def process_job_with_cleanup(job_id: str, urls: List[str], video_limit: int, llm_provider: str, analysis_results: dict):
    """Process a job without blocking - just start the pipeline"""
    try:
        # Start the pipeline - this will return immediately
        await process_creators_pipeline(job_id, urls, video_limit, llm_provider, analysis_results)
    except Exception as e:
        logger.error(f"Error starting job {job_id}: {str(e)}")
        # If we fail to start, clean up and start next job
        await start_next_job(analysis_results) 