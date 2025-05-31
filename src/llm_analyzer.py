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
from typing import Dict, Any, List
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAnalyzer:
    # Configuration constants
    MIN_VIOLATION_SCORE = 0.0  # Minimum score to consider a violation serious enough to report
    
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
        
        # Define how many tokens to reserve for the LLM's response
        self.response_token_reservation = int(os.getenv("LLM_RESPONSE_TOKEN_RESERVATION", "3000"))
        
        # Calculate max input tokens for the prompt (total context - reserved for response)
        # Ensure there's at least a minimum viable number of tokens for input (e.g., 1000)
        self.max_input_tokens = max(1000, self.context_size - self.response_token_reservation)
        
        logger.info(f"LLM Configuration - Model: {self.openai_model}, Context Size: {self.context_size}, Max Input Tokens (for prompt): {self.max_input_tokens}, Reserved Output Tokens (for response): {self.response_token_reservation}")
        
        # Handle default categories file path
        resolved_categories_file_path = categories_file
        if resolved_categories_file_path is None:
            src_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(src_dir)
            resolved_categories_file_path = os.path.join(project_root, "data", "YouTube_Controversy_Categories.csv")
            logger.info(f"LLMAnalyzer: No categories_file provided, defaulting to: {resolved_categories_file_path}")
        else:
            logger.info(f"LLMAnalyzer: Using provided categories_file: {resolved_categories_file_path}")

        try:
            self.categories_df = pd.read_csv(resolved_categories_file_path)
            if 'Category' not in self.categories_df.columns:
                logger.error(f"LLMAnalyzer: 'Category' column not found in {resolved_categories_file_path}. Category list will be empty.")
                # Create an empty DataFrame with 'Category' column to prevent errors in get_category_names
                self.categories_df = pd.DataFrame({'Category': pd.Series(dtype='str')})
            elif self.categories_df['Category'].isnull().all():
                logger.warning(f"LLMAnalyzer: 'Category' column in {resolved_categories_file_path} is present but all values are null/empty. Category list will be empty.")
                self.categories_df = pd.DataFrame({'Category': pd.Series(dtype='str')})
            else:
                # Drop rows where 'Category' is NaN, if any, and convert to string
                self.categories_df.dropna(subset=['Category'], inplace=True)
                self.categories_df['Category'] = self.categories_df['Category'].astype(str)
                logger.info(f"LLMAnalyzer: Successfully loaded {len(self.categories_df)} categories from {resolved_categories_file_path}. First few: {self.categories_df['Category'].head().tolist()}")
        except FileNotFoundError:
            logger.error(f"LLMAnalyzer: Categories CSV file not found at {resolved_categories_file_path}. Category list will be empty.")
            self.categories_df = pd.DataFrame({'Category': pd.Series(dtype='str')}) # Empty DataFrame with 'Category' column
        except pd.errors.EmptyDataError:
            logger.error(f"LLMAnalyzer: Categories CSV file at {resolved_categories_file_path} is empty. Category list will be empty.")
            self.categories_df = pd.DataFrame({'Category': pd.Series(dtype='str')})
        except Exception as e:
            logger.error(f"LLMAnalyzer: Error loading categories CSV from {resolved_categories_file_path}: {e}. Category list will be empty.")
            self.categories_df = pd.DataFrame({'Category': pd.Series(dtype='str')})

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
            
    def get_category_names(self) -> List[str]:
        """Returns the list of category names loaded from the CSV."""
        if self.categories_df is not None and 'Category' in self.categories_df.columns:
            return self.categories_df['Category'].tolist()
        return [] # Return empty list if not loaded properly

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
        base_prompt = f"""
You are a content compliance analyst tasked with identifying **clear and serious** violations in YouTube video transcripts. Your goal is to provide **detailed, conservative scores** for each compliance category.

---

### SCORING SCALE
Assign a score between 0.0 and 1.0 for each category:
- **0.0**: No violation
- **0.1–0.2**: Very minor concerns
- **0.3–0.5**: Moderate, clear violations
- **0.6–0.8**: Serious, repeated, or overtly harmful violations
- **0.9–1.0**: Egregious, dangerous, or extreme violations

---

### EVALUATION PRINCIPLES
- Focus on **serious, harmful, or illegal** content.
- Ignore casual profanity, normal banter, comedy, or minor infractions.
- Context and intent matter. Be lenient with educational or entertainment content.
- Only flag targeted, aggressive, explicit, or repeated problematic content.

---

### CONTENT GUIDANCE
- Do **not** penalize:
  - Reaction/commentary videos mentioning sensitive topics
  - Comedy, lifestyle, or gaming content with minor edgy language
  - Educational discussions of controversial issues
- Only flag:
  - Harmful misinformation
  - Repeated hate speech or dangerous behavior
  - Instructional or promotional illegal activity
  - Explicit or abusive content

---

### VIDEO TITLE
{video_title}

### CATEGORIES TO SCORE
{categories}

### TRANSCRIPT
PLACEHOLDER_FOR_TRANSCRIPT

---

### OUTPUT FORMAT (Always include every category, even if score is 0.0)
Respond with a valid JSON object in the following format:
{{
  "Category Name": {{
    "score": 0.0,
    "justification": "Brief reasoning for the score",
    "evidence": ["Example quote 1", "Example quote 2"]
  }},
  ...
}}

All categories must be included in the JSON response. If no violations exist, score should be 0.0 with minimal justification.

---
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
        # Strip common non-JSON prefixes/suffixes like backticks and language specifiers
        original_text = text # Keep original for logging if all fails
        text = re.sub(r"^```json\s*", "", text.strip()) # Remove ```json prefix
        text = re.sub(r"\s*```$", "", text) # Remove ``` suffix
        text = text.strip() # General strip

        try:
            # First attempt: Try to parse the whole text as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # Second attempt: Find what looks like the main JSON object using regex
            try:
                json_match = re.search(r'^\s*({[\s\S]*})\s*$', text) # Match if text IS a JSON object
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
            except json.JSONDecodeError: # Changed from generic except
                pass # Continue to next fallback
            except Exception as e: # Catch other regex/string errors
                logger.debug(f"_extract_valid_json: Regex search or inner parse failed: {e}")
                pass

            # Third attempt: Fallback for incomplete JSON (if it's just missing the final '}')
            if text.startswith('{') and not text.endswith('}'):
                try:
                    logger.warning("_extract_valid_json: Attempting to fix potentially incomplete JSON by adding closing brace.")
                    return json.loads(text + '}')
                except json.JSONDecodeError:
                    pass # If adding '}' doesn't work, proceed to manual extraction
            
            # Fourth attempt: Manual category extraction (as a last resort for very broken JSON)
            try:
                logger.warning("_extract_valid_json: Falling back to manual category extraction.")
                categories = {}
                pattern = r'"([^"]+)":\s*{([^{}]|{[^{}]*})*}' # This pattern tries to find "Category Name": { ... }
                matches = re.findall(pattern, original_text) # Use original_text for this fragile regex
                
                for match in matches:
                    category_name = match[0]
                    # The content captured by ([^{}]|{[^{}]*})* might be tricky.
                    # It's better to parse each identified segment carefully.
                    try:
                        # Reconstruct the potential JSON for the category item
                        # This is still risky if the content itself is malformed.
                        # Minimal assumption: structure is "CategoryName": { ... valid json ... }
                        # We need to find the full extent of the object for this category_name
                        # This regex is not perfect for that. A full parser would be better if this is common.
                        # For now, let's assume the regex found a reasonable segment for the category.
                        category_data_str = "{" + match[1] + "}"
                        category_data = json.loads(category_data_str)
                        categories[category_name] = category_data
                    except json.JSONDecodeError as e_cat:
                        logger.warning(f"_extract_valid_json: Manual extraction failed for category '{category_name}' content '{match[1]}': {e_cat}")
                        pass # Skip this category if its content is malformed
                
                if categories: # If we successfully extracted at least one category this way
                    logger.warning(f"_extract_valid_json: Manually extracted {len(categories)} categories.")
                    return categories
            except Exception as e_manual:
                logger.error(f"_extract_valid_json: Error during manual category extraction: {e_manual}")
                pass
                
        # All attempts failed
        logger.error(f"Failed to parse LLM response. Cleaned text for parsing: '{text}'. Raw text: '{original_text}', returning empty result")
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
                
                # CRITICAL FIX: Add debugging to track what the LLM actually returned
                logger.info(f"🤖 LLM returned {len(analysis_results)} categories for video {video_id}")
                for category, data in analysis_results.items():
                    score = data.get('score', 0)
                    logger.debug(f"   📊 '{category}': score {score}")
                
                # Filter out categories with low scores - only serious violations (>MIN_VIOLATION_SCORE)
                original_count = len(analysis_results)
                analysis_results = {
                    category: data for category, data in analysis_results.items()
                    if data.get('score', 0) > self.MIN_VIOLATION_SCORE
                }
                filtered_count = len(analysis_results)
                
                logger.info(f"🔍 Filtered results: {original_count} -> {filtered_count} categories (threshold: {self.MIN_VIOLATION_SCORE})")
                if filtered_count < original_count:
                    logger.debug(f"   🗑️ Filtered out {original_count - filtered_count} categories with scores <= {self.MIN_VIOLATION_SCORE}")
                
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
            "max_tokens": self.response_token_reservation
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
                
                # CRITICAL FIX: Add debugging to track what the LLM actually returned
                logger.info(f"🤖 OpenAI returned {len(analysis_results)} categories for video {video_id}")
                for category, data in analysis_results.items():
                    score = data.get('score', 0)
                    logger.debug(f"   📊 '{category}': score {score}")
                
                # Filter out categories with low scores - only serious violations (>MIN_VIOLATION_SCORE)
                original_count = len(analysis_results)
                analysis_results = {
                    category: data for category, data in analysis_results.items()
                    if data.get('score', 0) > self.MIN_VIOLATION_SCORE
                }
                filtered_count = len(analysis_results)
                
                logger.info(f"🔍 Filtered OpenAI results: {original_count} -> {filtered_count} categories (threshold: {self.MIN_VIOLATION_SCORE})")
                if filtered_count < original_count:
                    logger.debug(f"   🗑️ Filtered out {original_count - filtered_count} categories with scores <= {self.MIN_VIOLATION_SCORE}")
                
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
            "max_tokens": self.response_token_reservation
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
                            
                            # CRITICAL FIX: Add debugging to track what the LLM actually returned
                            logger.info(f"🤖 OpenAI returned {len(analysis_results)} categories for video {video_id}")
                            for category, data in analysis_results.items():
                                score = data.get('score', 0)
                                logger.debug(f"   📊 '{category}': score {score}")
                            
                            # Filter out categories with low scores - only serious violations (>MIN_VIOLATION_SCORE)
                            original_count = len(analysis_results)
                            analysis_results = {
                                category: data for category, data in analysis_results.items()
                                if data.get('score', 0) > self.MIN_VIOLATION_SCORE
                            }
                            filtered_count = len(analysis_results)
                            
                            logger.info(f"🔍 Filtered OpenAI results: {original_count} -> {filtered_count} categories (threshold: {self.MIN_VIOLATION_SCORE})")
                            if filtered_count < original_count:
                                logger.debug(f"   🗑️ Filtered out {original_count - filtered_count} categories with scores <= {self.MIN_VIOLATION_SCORE}")
                            
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
            predefined_categories = self.categories_df['Category'].tolist()

            for video_analysis in results["video_analyses"]:
                analysis_results_data = video_analysis.get("analysis", {}).get("results", {})
                for llm_category_name, llm_category_data in analysis_results_data.items():
                    # Find the matching predefined_category using fuzzy matching
                    matched_category_name = None
                    normalized_llm_category_name = llm_category_name.lower().replace('"', '').replace("'", "").strip()

                    for predefined_cat_name in predefined_categories:
                        normalized_predefined_cat_name = predefined_cat_name.lower().replace('"', '').replace("'", "").strip()
                        if normalized_llm_category_name == normalized_predefined_cat_name:
                            matched_category_name = predefined_cat_name
                            break
                    
                    if not matched_category_name:
                        # If no exact or fuzzy match, we might decide to log or skip
                        # For now, let's use the LLM's category name if no match,
                        # or one could choose to only include matched categories.
                        # This behavior should align with CreatorProcessor and export_handlers.
                        # For consistency with CreatorProcessor which uses LLM's categories as keys:
                        matched_category_name = llm_category_name 
                        if llm_category_name not in predefined_categories:
                             logger.warning(f"LLMAnalyzer summary: LLM category '{llm_category_name}' not in predefined list. Adding it to summary anyway.")


                    if matched_category_name not in summary:
                        summary[matched_category_name] = {
                            "max_score": 0.0,
                            "total_score": 0.0, # Will be used for average calculation
                            "videos_with_violations": 0,
                            "total_videos_for_category": 0, # Videos where this category was found
                            "examples": []
                        }
                    
                    summary[matched_category_name]["total_videos_for_category"] += 1
                    score = llm_category_data.get("score", 0)

                    if score > self.MIN_VIOLATION_SCORE: # Use MIN_VIOLATION_SCORE
                        summary[matched_category_name]["videos_with_violations"] += 1
                        summary[matched_category_name]["max_score"] = max(summary[matched_category_name]["max_score"], score)
                        summary[matched_category_name]["total_score"] += score
                        
                        summary[matched_category_name]["examples"].append({
                            "video_id": video_analysis["video_id"],
                            "video_title": video_analysis["video_title"],
                            "video_url": video_analysis["video_url"],
                            "score": score,
                            "evidence": llm_category_data.get("evidence", [])[0] if llm_category_data.get("evidence") else ""
                        })

            # Calculate averages and sort examples, and set overall total_videos
            total_analyzed_videos = len(results["video_analyses"])
            for category_name, cat_data in summary.items():
                if cat_data["videos_with_violations"] > 0:
                    cat_data["average_score"] = round(cat_data["total_score"] / cat_data["videos_with_violations"], 2)
                else:
                    cat_data["average_score"] = 0.0
                
                cat_data["examples"] = sorted(
                    cat_data["examples"],
                    key=lambda x: x.get("score", 0),
                    reverse=True
                )[:5]
                del cat_data["total_score"] # Remove temporary field
                cat_data["total_videos_in_channel"] = total_analyzed_videos # Add total videos analyzed in channel

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
            predefined_categories = self.categories_df['Category'].tolist()

            for video_analysis in results["video_analyses"]:
                analysis_results_data = video_analysis.get("analysis", {}).get("results", {})
                for llm_category_name, llm_category_data in analysis_results_data.items():
                    # Find the matching predefined_category using fuzzy matching
                    matched_category_name = None
                    normalized_llm_category_name = llm_category_name.lower().replace('"', '').replace("'", "").strip()

                    for predefined_cat_name in predefined_categories:
                        normalized_predefined_cat_name = predefined_cat_name.lower().replace('"', '').replace("'", "").strip()
                        if normalized_llm_category_name == normalized_predefined_cat_name:
                            matched_category_name = predefined_cat_name
                            break
                    
                    if not matched_category_name:
                        # If no exact or fuzzy match, we might decide to log or skip
                        # For now, let's use the LLM's category name if no match,
                        # or one could choose to only include matched categories.
                        # This behavior should align with CreatorProcessor and export_handlers.
                        # For consistency with CreatorProcessor which uses LLM's categories as keys:
                        matched_category_name = llm_category_name 
                        if llm_category_name not in predefined_categories:
                             logger.warning(f"LLMAnalyzer summary: LLM category '{llm_category_name}' not in predefined list. Adding it to summary anyway.")


                    if matched_category_name not in summary:
                        summary[matched_category_name] = {
                            "max_score": 0.0,
                            "total_score": 0.0, # Will be used for average calculation
                            "videos_with_violations": 0,
                            "total_videos_for_category": 0, # Videos where this category was found
                            "examples": []
                        }
                    
                    summary[matched_category_name]["total_videos_for_category"] += 1
                    score = llm_category_data.get("score", 0)

                    if score > self.MIN_VIOLATION_SCORE: # Use MIN_VIOLATION_SCORE
                        summary[matched_category_name]["videos_with_violations"] += 1
                        summary[matched_category_name]["max_score"] = max(summary[matched_category_name]["max_score"], score)
                        summary[matched_category_name]["total_score"] += score
                        
                        summary[matched_category_name]["examples"].append({
                            "video_id": video_analysis["video_id"],
                            "video_title": video_analysis["video_title"],
                            "video_url": video_analysis["video_url"],
                            "score": score,
                            "evidence": llm_category_data.get("evidence", [])[0] if llm_category_data.get("evidence") else ""
                        })

            # Calculate averages and sort examples, and set overall total_videos
            total_analyzed_videos = len(results["video_analyses"])
            for category_name, cat_data in summary.items():
                if cat_data["videos_with_violations"] > 0:
                    cat_data["average_score"] = round(cat_data["total_score"] / cat_data["videos_with_violations"], 2)
                else:
                    cat_data["average_score"] = 0.0
                
                cat_data["examples"] = sorted(
                    cat_data["examples"],
                    key=lambda x: x.get("score", 0),
                    reverse=True
                )[:5]
                del cat_data["total_score"] # Remove temporary field
                cat_data["total_videos_in_channel"] = total_analyzed_videos # Add total videos analyzed in channel

            results["summary"] = summary
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing channel content: {str(e)}")
            raise

    async def analyze_video_content_async(self, video_data):
        """Analyze content from a single video using async processing"""
        try:
            if not video_data.get('transcript'):
                return {
                    "video_id": video_data.get('id', 'unknown'),
                    "message": "No transcript available for this video.",
                    "results": {}
                }
            
            # Analyze the video
            analysis = await self.analyze_transcript_async(
                transcript_text=video_data['transcript']['full_text'],
                video_title=video_data.get('title', 'Unknown Title'),
                video_id=video_data.get('id', 'unknown')
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing video content: {str(e)}")
            return {
                "video_id": video_data.get('id', 'unknown'),
                "error": str(e),
                "results": {}
            }

    async def check_controversy_async(self, prompt: str) -> Dict[str, Any]:
        """
        Check for controversies using LLM (asynchronous version)
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Dictionary with controversy check results
        """
        try:
            if self.provider == "local":
                # For local LLM, use synchronous method
                return self._query_local_llm_simple(prompt)
            else:
                # For OpenAI, use async method
                return await self._query_openai_simple_async(prompt)
        except Exception as e:
            logger.error(f"Error checking controversy: {str(e)}")
            return {"is_controversial": False, "reason": "Check failed"}
    
    def _query_local_llm_simple(self, prompt: str) -> Dict[str, Any]:
        """Query local LLM for simple JSON responses"""
        url = f"{self.api_base_url}/chat/completions"
        
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Extract JSON from response
            return self._extract_valid_json(content)
        except Exception as e:
            logger.error(f"Error querying local LLM: {str(e)}")
            return {"is_controversial": False, "reason": "Check failed"}
    
    async def _query_openai_simple_async(self, prompt: str) -> Dict[str, Any]:
        """Query OpenAI for simple JSON responses"""
        url = f"{self.api_base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.openai_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200  # Small response expected
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    
                    result = await response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                    
                    # Extract JSON from response
                    return self._extract_valid_json(content)
        except Exception as e:
            logger.error(f"Error querying OpenAI: {str(e)}")
            return {"is_controversial": False, "reason": "Check failed"} 