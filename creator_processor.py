"""
Pipeline-based creator processing with efficient resource management
"""
import asyncio
import logging
import time
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from youtube_analyzer import YouTubeAnalyzer
from llm_analyzer import LLMAnalyzer
import statistics

# Get the logger for this module
logger = logging.getLogger('creator_processor')

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('creator_processor.log')
    ]
)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

@dataclass
class ProcessingConfig:
    """Configuration for creator processing"""
    max_concurrent_transcripts: int = 50
    max_concurrent_llm: int = 20
    batch_size: int = 10
    batch_delay: float = 0.0

class CreatorProcessor:
    def __init__(self, youtube_analyzer: YouTubeAnalyzer, llm_analyzer: LLMAnalyzer, config: Optional[ProcessingConfig] = None):
        self.youtube_analyzer = youtube_analyzer
        self.llm_analyzer = llm_analyzer
        self.config = config or ProcessingConfig()
        
        # Global semaphores for resource management
        self.transcript_semaphore = asyncio.Semaphore(self.config.max_concurrent_transcripts)
        self.llm_semaphore = asyncio.Semaphore(self.config.max_concurrent_llm)
        
        # Queues for pipeline stages
        self.transcript_queue = asyncio.Queue(maxsize=1000)
        self.llm_queue = asyncio.Queue(maxsize=1000)
        self.result_queue = asyncio.Queue(maxsize=1000)
        
        # Track active tasks and errors
        self.active_tasks = set()
        self.errors = {}
        
        # Track results by creator
        self.creator_results = {}
        
        # Timing metrics
        self.timing_metrics = {
            'channel_fetch': [],
            'transcript_fetch': [],
            'llm_analysis': [],
            'total_processing': []
        }
        self.start_time = None
        
        # Processing state
        self.processing_complete = asyncio.Event()
        self.total_creators = 0
        self.processed_creators = 0
        self.total_videos = 0
        self.processed_videos = 0
        
        logger.info("CreatorProcessor initialized with config: %s", self.config)
        
    def _calculate_percentiles(self, times: List[float]) -> Dict[str, float]:
        """Calculate p1, p50, p99 percentiles for a list of times"""
        if not times:
            return {'p1': 0, 'p50': 0, 'p99': 0}
        return {
            'p1': statistics.quantiles(times, n=100)[0],
            'p50': statistics.quantiles(times, n=100)[49],
            'p99': statistics.quantiles(times, n=100)[98]
        }
        
    def _log_completion_stats(self, results: List[Dict[str, Any]]) -> None:
        """Log completion statistics"""
        try:
            total_time = time.time() - self.start_time
            timing_stats = {
                'channel_fetch': self._calculate_percentiles(self.timing_metrics['channel_fetch']),
                'transcript_fetch': self._calculate_percentiles(self.timing_metrics['transcript_fetch']),
                'llm_analysis': self._calculate_percentiles(self.timing_metrics['llm_analysis'])
            }
            
            # Calculate statistics
            total_creators = len(results)
            successful_creators = sum(1 for r in results if r.get('status') == 'success')
            failed_creators = total_creators - successful_creators
            
            total_videos = sum(len(r.get('video_analyses', [])) for r in results)
            videos_with_transcripts = sum(
                1 for r in results 
                for v in r.get('video_analyses', [])
                if v.get('analysis', {}).get('results')
            )
            videos_without_transcripts = total_videos - videos_with_transcripts
            
            # Group errors by type
            error_types = {}
            for url, error in self.errors.items():
                error_type = error.split(':')[0] if ':' in error else error
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Log completion statistics
            logger.info("\n=== Bulk Processing Complete ===")
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"\nTiming Statistics (seconds):")
            for operation, stats in timing_stats.items():
                logger.info(f"\n{operation.replace('_', ' ').title()}:")
                logger.info(f"  P1:  {stats['p1']:.2f}")
                logger.info(f"  P50: {stats['p50']:.2f}")
                logger.info(f"  P99: {stats['p99']:.2f}")
            
            logger.info(f"\nCreator Statistics:")
            logger.info(f"  Total creators processed: {total_creators}")
            logger.info(f"  Successful creators: {successful_creators}")
            logger.info(f"  Failed creators: {failed_creators}")
            
            logger.info(f"\nVideo Statistics:")
            logger.info(f"  Total videos found: {total_videos}")
            logger.info(f"  Videos with transcripts: {videos_with_transcripts}")
            logger.info(f"  Videos without transcripts: {videos_without_transcripts}")
            
            if error_types:
                logger.info(f"\nError Breakdown:")
                for error_type, count in error_types.items():
                    logger.info(f"  - {error_type}: {count} occurrences")
            
            logger.info("\n=============================\n")
            
            # Force flush the logs
            for handler in logger.handlers:
                handler.flush()
                
        except Exception as e:
            logger.error(f"Error logging completion stats: {str(e)}", exc_info=True)
        
    async def _fetch_channel_data(self, url: str, video_limit: int) -> Dict[str, Any]:
        """Fetch channel data and queue videos for processing"""
        start_time = time.time()
        try:
            channel_data = await self.youtube_analyzer.analyze_channel_async(
                url,
                video_limit=video_limit,
                use_concurrent=True
            )
            
            if not channel_data:
                self.errors[url] = "Could not extract channel data"
                return None
                
            # Initialize results tracking for this creator
            self.creator_results[url] = {
                "url": url,
                "channel_id": channel_data.get('channel_id', "unknown"),
                "channel_name": channel_data.get('channel_name', "Unknown"),
                "channel_handle": channel_data.get('channel_handle', "Unknown"),
                "video_analyses": [],
                "expected_videos": len(channel_data.get('videos', [])),
                "processed_videos": 0,
                "status": "processing"
            }
            
            # Queue all videos for processing immediately
            for video in channel_data.get('videos', []):
                try:
                    await self.transcript_queue.put({
                        'video': video,
                        'channel_data': channel_data,
                        'creator_url': url
                    })
                    self.total_videos += 1
                except Exception as e:
                    logger.error(f"Error queueing video {video.get('id')} for {url}: {str(e)}")
                    continue
            
            # Record timing
            self.timing_metrics['channel_fetch'].append(time.time() - start_time)
            return channel_data
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error fetching channel data for {url}: {error_msg}")
            self.errors[url] = error_msg
            return None
    
    async def _process_transcripts(self):
        """Worker for processing transcripts"""
        while True:
            try:
                # Get next video to process
                data = await self.transcript_queue.get()
                video = data['video']
                creator_url = data['creator_url']
                
                start_time = time.time()
                async with self.transcript_semaphore:
                    # Get transcript
                    transcript = await self.youtube_analyzer.get_transcript_async(video['id'])
                    if transcript:
                        # Queue for LLM processing immediately
                        await self.llm_queue.put({
                            'video': video,
                            'transcript': transcript,
                            'creator_url': creator_url
                        })
                    else:
                        # Queue empty result immediately
                        await self.result_queue.put({
                            "video_id": video.get('id', 'unknown'),
                            "video_title": video.get('title', 'Unknown Title'),
                            "video_url": video.get('url', ''),
                            "creator_url": creator_url,
                            "analysis": {
                                "video_id": video.get('id', 'unknown'),
                                "message": "No transcript available for this video.",
                                "results": {}
                            }
                        })
                        self.processed_videos += 1
                
                # Record timing
                self.timing_metrics['transcript_fetch'].append(time.time() - start_time)
                self.transcript_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in transcript processing: {str(e)}")
                self.transcript_queue.task_done()
    
    async def _process_llm(self):
        """Worker for processing LLM analysis"""
        while True:
            try:
                # Get next video to analyze
                data = await self.llm_queue.get()
                video = data['video']
                transcript = data['transcript']
                creator_url = data['creator_url']
                
                start_time = time.time()
                async with self.llm_semaphore:
                    # Analyze with LLM
                    analysis = await self.llm_analyzer.analyze_transcript_async(
                        transcript_text=transcript['full_text'],
                        video_title=video.get('title', 'Unknown Title'),
                        video_id=video.get('id', 'unknown')
                    )
                    
                    # Queue result immediately
                    await self.result_queue.put({
                        "video_id": video.get('id', 'unknown'),
                        "video_title": video.get('title', 'Unknown Title'),
                        "video_url": video.get('url', ''),
                        "creator_url": creator_url,
                        "analysis": analysis
                    })
                    self.processed_videos += 1
                
                # Record timing
                self.timing_metrics['llm_analysis'].append(time.time() - start_time)
                self.llm_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in LLM processing: {str(e)}")
                self.llm_queue.task_done()
    
    async def _process_results(self):
        """Process results as they come in"""
        while True:
            try:
                result = await self.result_queue.get()
                creator_url = result['creator_url']
                
                if creator_url in self.creator_results:
                    creator_data = self.creator_results[creator_url]
                    creator_data['video_analyses'].append(result)
                    creator_data['processed_videos'] += 1
                    
                    # Check if all videos for this creator are processed
                    if creator_data['processed_videos'] >= creator_data['expected_videos']:
                        # Create summary and mark as complete
                        creator_data['summary'] = self._create_summary(creator_data['video_analyses'])
                        creator_data['status'] = 'success'
                        self.processed_creators += 1
                        
                        # Check if all creators are processed
                        if self.processed_creators >= self.total_creators:
                            self.processing_complete.set()
                
                self.result_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing results: {str(e)}")
                self.result_queue.task_done()
    
    async def process_creators(self, urls: List[str], video_limit: int) -> List[Dict[str, Any]]:
        """Process multiple creators through the pipeline"""
        logger.info("Starting bulk processing...")
        self.start_time = time.time()
        
        # Handle duplicate URLs
        unique_urls = []
        seen_urls = set()
        for url in urls:
            if url in seen_urls:
                logger.error(f"Duplicate URL found: {url}")
                self.errors[url] = "Duplicate URL - this channel was already processed"
            else:
                unique_urls.append(url)
                seen_urls.add(url)
        
        self.total_creators = len(unique_urls)
        self.processed_creators = 0
        self.total_videos = 0
        self.processed_videos = 0
        self.processing_complete.clear()
        
        # Start worker tasks
        transcript_workers = [
            asyncio.create_task(self._process_transcripts())
            for _ in range(self.config.max_concurrent_transcripts)
        ]
        llm_workers = [
            asyncio.create_task(self._process_llm())
            for _ in range(self.config.max_concurrent_llm)
        ]
        result_worker = asyncio.create_task(self._process_results())
        
        try:
            # Start fetching channel data for all creators immediately
            channel_tasks = []
            for url in unique_urls:
                task = asyncio.create_task(self._fetch_channel_data(url, video_limit))
                channel_tasks.append(task)
            
            # Wait for all channel data to be fetched
            await asyncio.gather(*channel_tasks)
            logger.info("All channel data fetched, waiting for processing to complete...")
            
            # Wait for all results to be processed
            await self.processing_complete.wait()
            logger.info("Processing complete, preparing final results...")
            
            # Convert results to list and handle any missing URLs
            results = []
            for url in urls:  # Use original urls list to maintain order and include duplicates
                if url in self.creator_results:
                    results.append(self.creator_results[url])
                else:
                    error_msg = self.errors.get(url, "Processing failed or timed out")
                    logger.error(f"Error processing URL {url}: {error_msg}")
                    results.append({
                        "url": url,
                        "error": error_msg,
                        "status": "error",
                        "channel_name": "Unknown",
                        "channel_handle": "Unknown",
                        "video_analyses": [],
                        "summary": {}
                    })
            
            # Log completion statistics
            logger.info("Logging completion statistics...")
            self._log_completion_stats(results)
            logger.info("Completion statistics logged.")
            
            return results
            
        finally:
            # Clean up worker tasks
            for worker in transcript_workers + llm_workers + [result_worker]:
                worker.cancel()
            logger.info("Worker tasks cleaned up.")
    
    def _create_summary(self, video_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary from video analyses"""
        summary = {}
        
        for analysis in video_analyses:
            results = analysis.get('analysis', {}).get('results', {})
            for category, data in results.items():
                if category not in summary:
                    summary[category] = {
                        "max_score": 0.0,
                        "total_score": 0.0,
                        "videos_with_violations": 0,
                        "total_videos": 0,
                        "examples": []
                    }
                
                # Update summary data
                summary[category]["total_videos"] += 1
                if data.get('score', 0) > 0:
                    summary[category]["videos_with_violations"] += 1
                    summary[category]["total_score"] += data['score']
                    summary[category]["max_score"] = max(
                        summary[category]["max_score"],
                        data['score']
                    )
                    
                    # Add example
                    summary[category]["examples"].append({
                        "video_id": analysis["video_id"],
                        "video_title": analysis["video_title"],
                        "video_url": analysis["video_url"],
                        "score": data['score'],
                        "evidence": data.get('evidence', [])[0] if data.get('evidence') else ""
                    })
        
        # Calculate averages and sort examples
        for category, data in summary.items():
            if data["videos_with_violations"] > 0:
                data["average_score"] = data["total_score"] / data["videos_with_violations"]
            else:
                data["average_score"] = 0.0
            
            # Sort examples by score
            data["examples"] = sorted(
                data["examples"],
                key=lambda x: x.get("score", 0),
                reverse=True
            )[:5]  # Keep only top 5 examples
            
            # Remove temporary total_score field
            del data["total_score"]
        
        return summary 