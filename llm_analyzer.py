"""
LLM Analyzer module for processing YouTube content against compliance categories
"""
import os
import json
import re
import requests
import pandas as pd
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAnalyzer:
    def __init__(self, provider="local", categories_file="YouTube_Controversy_Categories.csv"):
        """
        Initialize LLM analyzer with specified provider
        
        Args:
            provider: "local" or "openai"
            categories_file: Path to CSV file with controversy categories
        """
        self.provider = provider
        self.categories_df = pd.read_csv(categories_file)
        
        if provider == "local":
            self.api_base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
        elif provider == "openai":
            self.api_base_url = "https://api.openai.com/v1"
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not found in environment variables")
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
        
        prompt = f"""You are an objective content compliance analyzer.

TASK: Evaluate the following YouTube video transcript against each content compliance category. For each category, provide a score from 0 to 1, where:
- 0: No violation detected
- 0.25-0.5: Minor or ambiguous instances
- 0.75-1: Clear violations

VIDEO TITLE: {video_title}

CONTENT COMPLIANCE CATEGORIES:
{categories}

TRANSCRIPT:
{transcript_text[:8000]}  # Limiting transcript length to avoid context window issues

INSTRUCTIONS:
1. For EACH category listed above, provide:
   - Score (0-1)
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
        Analyze transcript content against compliance categories
        
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
    
    def _query_local_llm(self, prompt: str, video_id: str) -> Dict[str, Any]:
        """Query local Mistral LLM instance"""
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
        """Query OpenAI API"""
        url = f"{self.api_base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4",  # Can be made configurable
            "messages": [
                {"role": "system", "content": "You are a content compliance analyst that evaluates content against specific guidelines."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}  # Request JSON format from OpenAI
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
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
            logger.error(f"Error querying OpenAI: {str(e)}")
            return {"error": str(e)}
            
    def analyze_channel_content(self, channel_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze all videos from a channel
        
        Args:
            channel_data: Dictionary with channel videos and transcripts
            
        Returns:
            Dictionary with analysis results for each video
        """
        if not channel_data or 'videos' not in channel_data:
            return {"error": "Invalid channel data"}
            
        channel_analysis = {
            "channel_id": channel_data.get("channel_id", "unknown"),
            "video_analyses": [],
            "summary": {}
        }
        
        # Process each video
        for video in channel_data['videos']:
            if 'transcript' not in video:
                continue
                
            video_id = video['id']
            video_title = video['title']
            transcript_text = video['transcript']['full_text']
            
            # Analyze transcript
            analysis = self.analyze_transcript(transcript_text, video_title, video_id)
            
            channel_analysis['video_analyses'].append({
                "video_id": video_id,
                "video_title": video_title,
                "video_url": video['url'],
                "analysis": analysis
            })
            
        # Create channel-level summary
        channel_analysis['summary'] = self._create_channel_summary(channel_analysis['video_analyses'])
        
        return channel_analysis
    
    def _create_channel_summary(self, video_analyses: List[Dict]) -> Dict[str, Any]:
        """
        Create a summary of all video analyses for the channel
        
        Args:
            video_analyses: List of video analysis results
            
        Returns:
            Dictionary with category scores and evidence across all videos
        """
        # Get all category names from the CSV
        all_categories = self.categories_df['Category'].tolist()
        
        # Initialize summary structure
        summary = {
            category: {
                "max_score": 0.0,
                "average_score": 0.0,
                "videos_with_violations": 0,
                "total_videos": len(video_analyses),
                "examples": []
            } for category in all_categories
        }
        
        # Process each video analysis
        for video_analysis in video_analyses:
            analysis_results = video_analysis.get('analysis', {}).get('results', {})
            
            for category in all_categories:
                if category in analysis_results:
                    category_data = analysis_results[category]
                    score = category_data.get('score', 0.0)
                    
                    # Update summary data
                    if score > 0:
                        summary[category]['videos_with_violations'] += 1
                        
                        # Update max score
                        if score > summary[category]['max_score']:
                            summary[category]['max_score'] = score
                            
                        # Add example
                        if 'evidence' in category_data and category_data['evidence']:
                            summary[category]['examples'].append({
                                "video_id": video_analysis['video_id'],
                                "video_title": video_analysis['video_title'],
                                "video_url": video_analysis['video_url'],
                                "score": score,
                                "evidence": category_data['evidence'][0] if category_data['evidence'] else ""
                            })
        
        # Calculate average scores
        for category in all_categories:
            violations = summary[category]['videos_with_violations']
            if violations > 0:
                # Only count videos with violations in the average
                total_score = sum(example['score'] for example in summary[category]['examples'])
                summary[category]['average_score'] = total_score / violations
                
            # Sort examples by score (highest first)
            summary[category]['examples'] = sorted(
                summary[category]['examples'], 
                key=lambda x: x['score'], 
                reverse=True
            )[:3]  # Keep only top 3 examples
            
        return summary 