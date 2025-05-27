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

from src.youtube_analyzer import YouTubeAnalyzer
from src.llm_analyzer import LLMAnalyzer
from src.pipeline_workers import (
    channel_discovery_worker, controversy_screening_worker,
    video_transcript_worker, llm_worker, result_worker,
    monitor_pipeline_detailed, update_pipeline_stage
)

logger = logging.getLogger(__name__)

# Global job queue for sequential processing
job_queue = []
current_job_id = None
job_queue_lock = asyncio.Lock()

# Global task tracking for cancellation
active_job_tasks = {}  # job_id -> list of asyncio tasks

async def process_creators_pipeline(job_id: str, urls: List[str], video_limit: int, llm_provider: str, analysis_results: dict):
    """Queue-based pipeline with proper video-level rate limiting"""
    try:
        # Initialize timing statistics
        timing_stats = {
            'channel_discovery': [],
            'controversy_screening': [],
            'transcript_fetch': [],
            'llm_analysis': [],
            'result_processing': []
        }
        
        # Create queues for the pipeline
        channel_queue = asyncio.Queue(maxsize=1000)  # Channel URLs to discover videos
        controversy_queue = asyncio.Queue(maxsize=1000)  # Channels to screen for controversy
        video_queue = asyncio.Queue(maxsize=1000)    # Individual videos for transcript fetching
        llm_queue = asyncio.Queue(maxsize=1000)      # Videos with transcripts for LLM analysis
        result_queue = asyncio.Queue(maxsize=1000)   # LLM results for final processing
        
        # Initialize YouTube analyzer
        youtube_analyzer = YouTubeAnalyzer()
        
        # Initialize LLM analyzer
        actual_provider = os.getenv("LLM_PROVIDER", "local")
        llm_analyzer = LLMAnalyzer(provider=actual_provider)
        
        # Queue all URLs for channel discovery
        for url in urls:
            await channel_queue.put({
                'url': url,
                'video_limit': video_limit
            })
        
        # Track initial queue size
        initial_channel_size = channel_queue.qsize()
        logger.info(f"📊 Pipeline initialized with {initial_channel_size} channels to process")
        
        # Create workers for each stage
        workers = []
        
        # Channel discovery workers (2 workers)
        channel_workers = []
        for i in range(2):
            worker = asyncio.create_task(
                channel_discovery_worker(i, channel_queue, controversy_queue, youtube_analyzer, timing_stats, job_id, analysis_results)
            )
            workers.append(worker)
            channel_workers.append(worker)
        
        # Controversy screening workers (2 workers)
        controversy_workers = []
        for i in range(2):
            worker = asyncio.create_task(
                controversy_screening_worker(i, controversy_queue, video_queue, llm_analyzer, timing_stats, job_id, analysis_results)
            )
            workers.append(worker)
            controversy_workers.append(worker)
        
        # Video transcript workers (4 workers)
        transcript_workers = []
        for i in range(4):
            worker = asyncio.create_task(
                video_transcript_worker(i, video_queue, llm_queue, youtube_analyzer, timing_stats, job_id, analysis_results)
            )
            workers.append(worker)
            transcript_workers.append(worker)
        
        # LLM analysis workers (3 workers)
        llm_workers = []
        for i in range(3):
            worker = asyncio.create_task(
                llm_worker(i, llm_queue, result_queue, llm_analyzer, timing_stats, job_id, analysis_results)
            )
            workers.append(worker)
            llm_workers.append(worker)
        
        # Result processing worker (1 worker)
        result_worker_task = asyncio.create_task(
            result_worker(0, result_queue, timing_stats, job_id, analysis_results)
        )
        workers.append(result_worker_task)
        
        # Monitor pipeline progress
        monitor_task = asyncio.create_task(
            monitor_pipeline_detailed(job_id, channel_queue, controversy_queue, video_queue, llm_queue, result_queue, len(urls), timing_stats, analysis_results)
        )
        
        # Track all tasks for cancellation
        if job_id not in active_job_tasks:
            active_job_tasks[job_id] = []
        active_job_tasks[job_id].extend(workers + [monitor_task])
        
        # Wait for all queues to be processed
        await channel_queue.join()
        logger.info(f"✅ All channels discovered for job {job_id}")
        
        await controversy_queue.join()
        logger.info(f"✅ All controversy screening completed for job {job_id}")
        
        await video_queue.join()
        logger.info(f"✅ All video transcripts fetched for job {job_id}")
        
        await llm_queue.join()
        logger.info(f"✅ All LLM analyses completed for job {job_id}")
        
        await result_queue.join()
        logger.info(f"✅ All results processed for job {job_id}")
        
        # Cancel all workers
        for worker in workers:
            worker.cancel()
        monitor_task.cancel()
        
        # Wait for all cancelled tasks to finish
        await asyncio.gather(*workers, monitor_task, return_exceptions=True)
        
        # Wait a moment for final monitoring data
        await asyncio.sleep(0.5)
        
        # Calculate final statistics
        successful = len(analysis_results[job_id]['results'])
        failed = len(analysis_results[job_id]['failed_urls'])
        total_processed = successful + failed
        
        logger.info("=" * 100)
        logger.info(f"🏁 FINAL STATISTICS for job {job_id}")
        logger.info("=" * 100)
        logger.info(f"📊 PROCESSING SUMMARY:")
        logger.info(f"   └─ URLs submitted: {len(urls)}")
        logger.info(f"   └─ URLs processed: {total_processed}")
        logger.info(f"   └─ ✅ Successful: {successful}")
        logger.info(f"   └─ ❌ Failed: {failed}")
        
        # Mark job as complete
        analysis_results[job_id]['status'] = 'completed'
        analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
        analysis_results[job_id]['processed_urls'] = total_processed
        
        # Create channel summaries from individual video results
        await create_channel_summaries(job_id, analysis_results)
        
        # Clean up task tracking
        if job_id in active_job_tasks:
            del active_job_tasks[job_id]
        
    except Exception as e:
        logger.error(f"💥 Error in pipeline processing: {str(e)}")
        if job_id in analysis_results:
            analysis_results[job_id]['status'] = 'failed'
            analysis_results[job_id]['error'] = str(e)
            analysis_results[job_id]['completed_at'] = datetime.now().isoformat()

            # Cancel and clean up any running workers
            if 'workers' in locals():
                for worker in workers:
                    worker.cancel()
                if 'monitor_task' in locals():
                    monitor_task.cancel()
                await asyncio.gather(*workers, monitor_task, return_exceptions=True)
            
            # Clean up task tracking on failure
            if job_id in active_job_tasks:
                del active_job_tasks[job_id]

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
        
        # Process each channel's video results
        channels_to_remove = []  # Track channels that should be moved to failed
        
        for url, channel_data in analysis_results[job_id]['results'].items():
            # Check if this channel has any successfully analyzed videos
            video_analyses = channel_data.get('video_analyses', [])
            
            if not video_analyses:
                # No videos were successfully analyzed - this is a failure
                logger.warning(f"❌ Channel {url} has no successfully analyzed videos - moving to failed")
                
                # Add to failed URLs
                analysis_results[job_id]['failed_urls'].append({
                    'url': url,
                    'error': 'Failed to analyze any videos from this channel. All transcript downloads may have failed.',
                    'error_type': 'no_videos_analyzed',
                    'channel_name': channel_data.get('channel_name', 'Unknown'),
                    'channel_id': channel_data.get('channel_id', 'unknown'),
                    'video_count': len(channel_data.get('original_videos', []))
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
    """Process a job and handle queue cleanup when done"""
    try:
        await process_creators_pipeline(job_id, urls, video_limit, llm_provider, analysis_results)
    finally:
        # Clean up and start next job
        async with job_queue_lock:
            global current_job_id
            current_job_id = None
            
            # Start next job if any are queued
            if job_queue:
                next_job = job_queue.pop(0)
                current_job_id = next_job['job_id']
                
                # Update status and start processing
                analysis_results[current_job_id]['status'] = 'processing'
                analysis_results[current_job_id]['queue_position'] = 0
                
                logger.info(f"🚀 Starting next queued job {current_job_id}")
                
                task = asyncio.create_task(
                    process_job_with_cleanup(
                        current_job_id,
                        next_job['urls'],
                        next_job['video_limit'],
                        next_job['llm_provider'],
                        analysis_results
                    )
                )
                
                # Track task for cancellation
                active_job_tasks[current_job_id] = [task] 