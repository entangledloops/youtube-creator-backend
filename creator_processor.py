"""
Pipeline-based creator processing with efficient resource management
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from youtube_analyzer import YouTubeAnalyzer
from llm_analyzer import LLMAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for creator processing"""
    max_concurrent_transcripts: int = 10  # Max concurrent transcript fetches across all creators
    max_concurrent_llm: int = 5  # Max concurrent LLM requests across all creators
    batch_size: int = 3  # Number of creators to process simultaneously
    batch_delay: float = 1.0  # Delay between creator batches in seconds

class CreatorProcessor:
    def __init__(self, youtube_analyzer: YouTubeAnalyzer, llm_analyzer: LLMAnalyzer, config: Optional[ProcessingConfig] = None):
        self.youtube_analyzer = youtube_analyzer
        self.llm_analyzer = llm_analyzer
        self.config = config or ProcessingConfig()
        
        # Global semaphores for resource management
        self.transcript_semaphore = asyncio.Semaphore(self.config.max_concurrent_transcripts)
        self.llm_semaphore = asyncio.Semaphore(self.config.max_concurrent_llm)
        
        # Queues for pipeline stages
        self.transcript_queue = asyncio.Queue()
        self.llm_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        
        # Track active tasks
        self.active_tasks = set()
        
    async def process_creator(self, url: str, video_limit: int) -> Dict[str, Any]:
        """Process a single creator through the pipeline"""
        try:
            # Stage 1: Get channel data
            channel_data = await self.youtube_analyzer.analyze_channel_async(
                url,
                video_limit=video_limit,
                use_concurrent=True
            )
            
            if not channel_data:
                return {
                    "url": url,
                    "error": "Could not extract channel data"
                }
            
            # Stage 2: Queue videos for transcript processing
            for video in channel_data.get('videos', []):
                await self.transcript_queue.put({
                    'video': video,
                    'channel_data': channel_data
                })
            
            # Stage 3: Wait for all processing to complete
            results = []
            expected_videos = len(channel_data.get('videos', []))
            processed_videos = 0
            
            while processed_videos < expected_videos:
                result = await self.result_queue.get()
                results.append(result)
                processed_videos += 1
            
            # Stage 4: Combine results
            return {
                "url": url,
                "channel_id": channel_data.get('channel_id', "unknown"),
                "channel_name": channel_data.get('channel_name', "Unknown"),
                "channel_handle": channel_data.get('channel_handle', "Unknown"),
                "video_analyses": results,
                "summary": self._create_summary(results)
            }
            
        except Exception as e:
            logger.error(f"Error processing creator {url}: {str(e)}")
            return {
                "url": url,
                "error": str(e)
            }
    
    async def _process_transcripts(self):
        """Worker for processing transcripts"""
        while True:
            try:
                # Get next video to process
                data = await self.transcript_queue.get()
                video = data['video']
                channel_data = data['channel_data']
                
                async with self.transcript_semaphore:
                    # Get transcript
                    transcript = await self.youtube_analyzer.get_transcript_async(video['id'])
                    if transcript:
                        # Queue for LLM processing
                        await self.llm_queue.put({
                            'video': video,
                            'transcript': transcript,
                            'channel_data': channel_data
                        })
                    else:
                        # Queue empty result
                        await self.result_queue.put({
                            "video_id": video.get('id', 'unknown'),
                            "video_title": video.get('title', 'Unknown Title'),
                            "video_url": video.get('url', ''),
                            "analysis": {
                                "video_id": video.get('id', 'unknown'),
                                "message": "No transcript available for this video.",
                                "results": {}
                            }
                        })
                
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
                
                async with self.llm_semaphore:
                    # Analyze with LLM
                    analysis = await self.llm_analyzer.analyze_transcript_async(
                        transcript_text=transcript['full_text'],
                        video_title=video.get('title', 'Unknown Title'),
                        video_id=video.get('id', 'unknown')
                    )
                    
                    # Queue result
                    await self.result_queue.put({
                        "video_id": video.get('id', 'unknown'),
                        "video_title": video.get('title', 'Unknown Title'),
                        "video_url": video.get('url', ''),
                        "analysis": analysis
                    })
                
                self.llm_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in LLM processing: {str(e)}")
                self.llm_queue.task_done()
    
    async def process_creators(self, urls: List[str], video_limit: int) -> List[Dict[str, Any]]:
        """Process multiple creators through the pipeline"""
        # Start worker tasks
        transcript_workers = [
            asyncio.create_task(self._process_transcripts())
            for _ in range(self.config.max_concurrent_transcripts)
        ]
        llm_workers = [
            asyncio.create_task(self._process_llm())
            for _ in range(self.config.max_concurrent_llm)
        ]
        
        try:
            # Process creators in batches
            results = []
            for i in range(0, len(urls), self.config.batch_size):
                batch = urls[i:i + self.config.batch_size]
                batch_tasks = [
                    self.process_creator(url, video_limit)
                    for url in batch
                ]
                
                # Process batch
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)
                
                # Add delay between batches if not the last batch
                if i + self.config.batch_size < len(urls):
                    await asyncio.sleep(self.config.batch_delay)
            
            return results
            
        finally:
            # Clean up worker tasks
            for worker in transcript_workers + llm_workers:
                worker.cancel()
    
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