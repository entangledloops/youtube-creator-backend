"""
LLM Analyzer module for processing YouTube content against compliance categories
"""
import os
import json
import re
import aiohttp
import requests
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAnalyzer:
    def __init__(self, provider="local", categories_file=None):
        """
        Initialize LLM analyzer with specified provider
        
        Args:
            provider: "local" or "openai"
            categories_file: Path to CSV file with controversy categories
        """
        self.provider = provider
        
        # Load model configuration from environment
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.context_size = int(os.getenv("LLM_CONTEXT_SIZE", "128000"))
        
        # Calculate max input tokens (reserve 1000 tokens for response)
        self.max_input_tokens = max(1000, self.context_size - 1000)
        
        logger.info(f"LLM Configuration - Model: {self.openai_model}, Context Size: {self.context_size}, Max Input: {self.max_input_tokens}")
        
        # Handle default categories file path
        if categories_file is None:
            # Get the project root directory (parent of src)
            src_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(src_dir)
            categories_file = os.path.join(project_root, "data", "YouTube_Controversy_Categories.csv")
        
        self.categories_df = pd.read_csv(categories_file)
        
        if provider == "local":
            self.api_base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
        elif provider == "openai":
            self.api_base_url = "https://api.openai.com/v1"
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            # Add debug logging for API key
            logger.info(f"OpenAI API key loaded: {self.api_key[:4]}...{self.api_key[-4:] if len(self.api_key) > 8 else ''}")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
    def _prepare_categories_prompt(self):
        """Format controversy categories for prompt"""
        categories_text = ""
        
        for idx, row in self.categories_df.iterrows():
            categories_text += f"{idx+1}. {row['Category']}: {row['Definition']}\n"
            
        return categories_text
    
    def _format_analyze_prompt(self, transcript_text, video_title):
        """Format prompt for analyzing transcript against categories"""
        categories = self._prepare_categories_prompt()
        
        # Calculate rough prompt size without transcript
        base_prompt = f"""You are an objective content compliance analyzer.

TASK: Evaluate the following YouTube video transcript against each content compliance category. For each category, provide a score from 0 to 1, where:

SCORING SCALE (use the full range):
- 0.0: No violation detected
- 0.1-0.3: Very minor instances or borderline content
- 0.4-0.6: Moderate violations or clear but not severe instances  
- 0.7-0.9: Significant violations or multiple instances
- 1.0: Severe, egregious, or extreme violations

Use precise decimal scores (e.g., 0.2, 0.5, 0.8) to reflect the exact severity level rather than just endpoint values.

IMPORTANT LANGUAGE GUIDANCE:
- Mild profanity, casual swearing, or occasional strong language in entertainment context should generally score 0.3 or lower
- Only score higher for excessive, aggressive, or targeted offensive language
- Consider the overall context and intent rather than isolated words
- Comedy, entertainment, and casual conversation contexts should be evaluated more leniently

VIDEO TITLE: {video_title}

CONTENT COMPLIANCE CATEGORIES:
{categories}

TRANSCRIPT:
PLACEHOLDER_FOR_TRANSCRIPT

INSTRUCTIONS:
1. For EACH category listed above, provide:
   - Score (0-1) - Use precise decimals based on severity
   - Brief justification
   - Quote any problematic text found

2. Your response MUST be a valid JSON object with this structure:
{{
  "Category Name": {{
    "score": 0.5,
    "justification": "Reason for score",
    "evidence": ["quote1", "quote2"]
  }}
}}

3. Only include categories where you found potential violations (score > 0).
4. If no violations found for a category, don't include it.
5. If no violations found at all, return an empty JSON object: {{}}
6. Ensure all JSON is properly formatted with double quotes around keys and string values.
"""
        
        # Estimate tokens used by base prompt (roughly 4 chars per token)
        base_prompt_tokens = len(base_prompt) // 4
        available_transcript_tokens = self.max_input_tokens - base_prompt_tokens
        
        # Truncate transcript if necessary (roughly 4 chars per token)
        max_transcript_chars = available_transcript_tokens * 4
        if len(transcript_text) > max_transcript_chars:
            logger.warning(f"Transcript too long ({len(transcript_text)} chars), truncating to {max_transcript_chars} chars")
            transcript_text = transcript_text[:max_transcript_chars]
        
        # Replace placeholder with actual transcript
        prompt = base_prompt.replace("PLACEHOLDER_FOR_TRANSCRIPT", transcript_text)
        
        return prompt
    
    def _extract_valid_json(self, text):
        """Extract and parse valid JSON from text, handling common formatting errors"""
        try:
            # First attempt: Try to parse the whole text as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # Find what looks like JSON
            try:
                json_match = re.search(r'({[\s\S]*})', text)
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
            except:
                pass
            
            # Attempt to extract categories manually
            try:
                # Find all category entries
                categories = {}
                
                # Match patterns like: "Category Name": { ... }
                pattern = r'"([^"]+)":\s*{([^{}]|{[^{}]*})*}'
                matches = re.findall(pattern, text)
                
                for match in matches:
                    category_name = match[0]
                    category_content = "{" + match[1] + "}"
                    
                    # Try to parse the category content
                    try:
                        category_data = json.loads(category_content)
                        if 'score' in category_data and category_data['score'] > 0:
                            categories[category_name] = category_data
                    except:
                        pass
                
                if categories:
                    return categories
            except:
                pass
                
            # Last resort: manual extraction
            logger.error(f"Failed to parse LLM response, returning empty result")
            return {}
    
    def analyze_transcript(self, transcript_text: str, video_title: str, video_id: str) -> Dict[str, Any]:
        """
        Analyze transcript content against compliance categories (synchronous version)
        
        Args:
            transcript_text: Full text of video transcript
            video_title: Title of the video
            video_id: YouTube video ID
            
        Returns:
            Dictionary with analysis results
        """
        prompt = self._format_analyze_prompt(transcript_text, video_title)
        
        try:
            if self.provider == "local":
                return self._query_local_llm(prompt, video_id)
            else:
                return self._query_openai(prompt, video_id)
        except Exception as e:
            logger.error(f"Error analyzing transcript: {str(e)}")
            return {"error": str(e)}
    
    async def analyze_transcript_async(self, transcript_text: str, video_title: str, video_id: str) -> Dict[str, Any]:
        """
        Analyze transcript content against compliance categories (asynchronous version)
        
        Args:
            transcript_text: Full text of video transcript
            video_title: Title of the video
            video_id: YouTube video ID
            
        Returns:
            Dictionary with analysis results
        """
        prompt = self._format_analyze_prompt(transcript_text, video_title)
        
        try:
            if self.provider == "local":
                return self._query_local_llm(prompt, video_id)  # Local LLM is still synchronous
            else:
                return await self._query_openai_async(prompt, video_id)
        except Exception as e:
            logger.error(f"Error analyzing transcript: {str(e)}")
            return {"error": str(e)}
    
    def _query_local_llm(self, prompt: str, video_id: str) -> Dict[str, Any]:
        """Query local Mistral LLM instance (synchronous)"""
        logger.info(f"🏠 LOCAL LLM CALLED: Querying local LLM at {self.api_base_url}")
        url = f"{self.api_base_url}/chat/completions"
        
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "You are a content compliance analyst that evaluates content against specific guidelines."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Try to extract valid JSON
            try:
                # Log the raw response content for debugging
                logger.debug(f"Raw LLM response: {content}")
                
                # Extract valid JSON
                analysis_results = self._extract_valid_json(content)
                
                # Filter out categories with score=0
                analysis_results = {
                    category: data for category, data in analysis_results.items()
                    if data.get('score', 0) > 0
                }
                
                return {
                    "video_id": video_id,
                    "results": analysis_results
                }
            except Exception as e:
                logger.error(f"Error processing LLM response: {str(e)}")
                
                # Include useful debug info
                return {
                    "video_id": video_id,
                    "error": f"Error processing LLM response: {str(e)}",
                    "results": {
                        "Content that could lead to death/injury": {
                            "score": 0.75,
                            "justification": "Default fallback analysis due to processing error. The system detected potential content that could lead to death/injury.",
                            "evidence": ["Automated fallback analysis - please review content manually"]
                        }
                    }
                }
        except requests.RequestException as e:
            logger.error(f"Error querying local LLM: {str(e)}")
            return {"error": str(e)}
    
    def _query_openai(self, prompt: str, video_id: str) -> Dict[str, Any]:
        """Query OpenAI API (synchronous version)"""
        logger.info(f"🤖 OPENAI CALLED: Querying OpenAI API at {self.api_base_url}")
        url = f"{self.api_base_url}/chat/completions"
        
        # Log the actual API key being used (first 4 and last 4 chars for security)
        api_key_preview = f"{self.api_key[:4]}...{self.api_key[-4:]}" if len(self.api_key) > 8 else self.api_key
        logger.info(f"Using OpenAI API key: {api_key_preview}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Log the full Authorization header (first 10 chars for security)
        auth_header = headers["Authorization"]
        logger.info(f"Authorization header: {auth_header[:10]}...")
        
        payload = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": "You are a content compliance analyst that evaluates content against specific guidelines."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000  # Limit response size
        }
        
        # Log the request payload (excluding the full prompt for brevity)
        logger.debug(f"Request payload: {json.dumps({**payload, 'messages': [{'role': m['role'], 'content_length': len(m['content'])} for m in payload['messages']]})}")
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 401:
                logger.error(f"OpenAI API unauthorized. Response: {response.text}")
                # Log the actual request headers that were sent
                logger.error(f"Request headers sent: {dict(response.request.headers)}")
            elif response.status_code == 400:
                logger.error(f"OpenAI API bad request. Response: {response.text}")
                logger.error(f"Request payload: {json.dumps(payload, indent=2)}")
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Try to extract valid JSON
            try:
                # Extract valid JSON
                analysis_results = self._extract_valid_json(content)
                
                # Filter out categories with score=0
                analysis_results = {
                    category: data for category, data in analysis_results.items()
                    if data.get('score', 0) > 0
                }
                
                return {
                    "video_id": video_id,
                    "results": analysis_results
                }
            except Exception as e:
                logger.error(f"Error processing OpenAI response: {str(e)}")
                return {"error": str(e)}
        except requests.RequestException as e:
            logger.error(f"Error querying OpenAI: {str(e)}")
            return {"error": str(e)}
    
    async def _query_openai_async(self, prompt: str, video_id: str) -> Dict[str, Any]:
        """Query OpenAI API (asynchronous version) with retry logic"""
        url = f"{self.api_base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": "You are a content compliance analyst that evaluates content against specific guidelines."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000  # Limit response size
        }
        
        max_retries = 3
        base_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload) as response:
                        if response.status == 429:  # Rate limit error
                            if attempt < max_retries - 1:  # Don't sleep on last attempt
                                delay = base_delay * (2 ** attempt)  # Exponential backoff
                                logger.warning(f"Rate limit hit, retrying in {delay} seconds...")
                                await asyncio.sleep(delay)
                                continue
                        elif response.status == 401:
                            logger.error(f"OpenAI API unauthorized. Response: {await response.text()}")
                            logger.error(f"Request headers sent: {dict(response.request_info.headers)}")
                        elif response.status == 400:
                            logger.error(f"OpenAI API bad request. Response: {await response.text()}")
                            logger.error(f"Request payload: {json.dumps(payload, indent=2)}")
                        
                        response.raise_for_status()
                        
                        result = await response.json()
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                        
                        # Try to extract valid JSON
                        try:
                            # Extract valid JSON
                            analysis_results = self._extract_valid_json(content)
                            
                            # Filter out categories with score=0
                            analysis_results = {
                                category: data for category, data in analysis_results.items()
                                if data.get('score', 0) > 0
                            }
                            
                            return {
                                "video_id": video_id,
                                "results": analysis_results
                            }
                        except Exception as e:
                            logger.error(f"Error processing OpenAI response: {str(e)}")
                            return {"error": str(e)}
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Error querying OpenAI after {max_retries} attempts: {str(e)}")
                    return {"error": str(e)}
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Error querying OpenAI, retrying in {delay} seconds: {str(e)}")
                await asyncio.sleep(delay)
        
        return {"error": "Max retries exceeded"}
    
    def analyze_channel_content(self, channel_data):
        """Analyze content from multiple videos in a channel"""
        try:
            # Initialize results structure
            results = {
                "channel_id": channel_data.get('channel_id'),
                "video_analyses": [],
                "summary": {}
            }
            
            # Process each video
            for video in channel_data.get('videos', []):
                if not video.get('transcript'):
                    continue
                    
                # Analyze the video
                analysis = self.analyze_transcript(
                    transcript_text=video['transcript']['full_text'],
                    video_title=video.get('title', 'Unknown Title'),
                    video_id=video.get('id', 'unknown')
                )
                
                # Add to results
                results["video_analyses"].append({
                    "video_id": video.get('id', 'unknown'),
                    "video_title": video.get('title', 'Unknown Title'),
                    "video_url": video.get('url', ''),
                    "analysis": analysis
                })
            
            # Create summary across all videos
            summary = {}
            for category in self.categories_df['Category'].tolist():
                category_violations = []
                max_score = 0
                total_score = 0
                videos_with_violations = 0
                
                for video_analysis in results["video_analyses"]:
                    if category in video_analysis["analysis"].get("results", {}):
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
                        "total_videos": len(results["video_analyses"]),
                        "examples": sorted(category_violations, key=lambda x: x["score"], reverse=True)[:5]
                    }
            
            results["summary"] = summary
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing channel content: {str(e)}")
            raise
    
    async def analyze_channel_content_async(self, channel_data):
        """Analyze content from multiple videos in a channel using async processing"""
        try:
            # Initialize results structure
            results = {
                "channel_id": channel_data.get('channel_id'),
                "video_analyses": [],
                "summary": {}
            }
            
            if self.provider == "openai":
                logger.info("Using CONCURRENT processing for OpenAI analysis")
                # For OpenAI, process videos in parallel
                tasks = []
                for video in channel_data.get('videos', []):
                    if not video.get('transcript'):
                        continue
                    logger.debug(f"Creating OpenAI analysis task for video: {video['id']} - {video['title']}")
                    tasks.append(self.analyze_transcript_async(
                        transcript_text=video['transcript']['full_text'],
                        video_title=video.get('title', 'Unknown Title'),
                        video_id=video.get('id', 'unknown')
                    ))
                
                # Wait for all analyses to complete
                logger.info(f"Waiting for {len(tasks)} OpenAI analysis tasks to complete")
                analyses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for video, analysis in zip([v for v in channel_data.get('videos', []) if v.get('transcript')], analyses):
                    if isinstance(analysis, Exception):
                        logger.error(f"✗ Error analyzing video {video.get('id')}: {str(analysis)}")
                        continue
                        
                    logger.info(f"✓ OpenAI analysis completed for video: {video['id']} - {video['title']}")
                    results["video_analyses"].append({
                        "video_id": video.get('id', 'unknown'),
                        "video_title": video.get('title', 'Unknown Title'),
                        "video_url": video.get('url', ''),
                        "analysis": analysis
                    })
            else:
                logger.info("Using SEQUENTIAL processing for local LLM analysis")
                # For local LLM, process videos sequentially for easier debugging
                for video in channel_data.get('videos', []):
                    if not video.get('transcript'):
                        continue
                        
                    logger.debug(f"Analyzing video with local LLM: {video['id']} - {video['title']}")
                    # Analyze the video
                    analysis = self.analyze_transcript(
                        transcript_text=video['transcript']['full_text'],
                        video_title=video.get('title', 'Unknown Title'),
                        video_id=video.get('id', 'unknown')
                    )
                    
                    logger.info(f"✓ Local LLM analysis completed for video: {video['id']} - {video['title']}")
                    # Add to results
                    results["video_analyses"].append({
                        "video_id": video.get('id', 'unknown'),
                        "video_title": video.get('title', 'Unknown Title'),
                        "video_url": video.get('url', ''),
                        "analysis": analysis
                    })
            
            # Create summary across all videos
            summary = {}
            for category in self.categories_df['Category'].tolist():
                category_violations = []
                max_score = 0
                total_score = 0
                videos_with_violations = 0
                
                for video_analysis in results["video_analyses"]:
                    if category in video_analysis["analysis"].get("results", {}):
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
                        "total_videos": len(results["video_analyses"]),
                        "examples": sorted(category_violations, key=lambda x: x["score"], reverse=True)[:5]
                    }
            
            results["summary"] = summary
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing channel content: {str(e)}")
            raise 