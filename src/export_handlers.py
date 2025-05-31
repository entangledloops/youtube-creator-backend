"""
Export handlers for CSV and evidence downloads
"""
import os
import json
import pandas as pd
from fastapi import HTTPException, Response
from fastapi.responses import FileResponse
import logging
import csv
import io
from typing import List, Dict, Any

from src.llm_analyzer import LLMAnalyzer # For category names

logger = logging.getLogger(__name__)

# Initialize LLMAnalyzer to get category names (consider passing as arg or centralizing)
# This instance is only used for get_category_names, so default init is fine.
llm_analyzer_for_categories = LLMAnalyzer()
all_categories_glob = llm_analyzer_for_categories.get_category_names() # Load once

# CRITICAL LOG: Check the loaded global categories immediately
if not all_categories_glob:
    logger.critical("EXPORT_HANDLERS: CRITICAL - Global category list (all_categories_glob) is EMPTY or FAILED TO LOAD. Overall scores will likely be 0.")
elif len(all_categories_glob) < 5: # Arbitrary small number to indicate potential partial load or issue
    logger.warning(f"EXPORT_HANDLERS: WARNING - Global category list (all_categories_glob) seems very short ({len(all_categories_glob)} categories): {all_categories_glob}. This might be an issue.")
else:
    logger.info(f"EXPORT_HANDLERS: Successfully loaded {len(all_categories_glob)} global categories. First few: {all_categories_glob[:5]}")

def get_output_directory() -> str:
    """Get the output directory from environment variable with default fallback"""
    return os.getenv("OUTPUT_DIR", "output")

async def download_bulk_analysis_csv(job_id: str, analysis_results: Dict[str, Any]):
    """Download bulk analysis results as CSV"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = analysis_results[job_id]
    
    # Allow access to partial results for cancelled jobs too
    if job_data['status'] == 'processing' and not job_data.get('partial_results_ok', False): # Add a flag if we want to allow download during processing
        raise HTTPException(status_code=400, detail="Analysis still in progress. Cannot download CSV yet.")

    # Use the globally loaded categories, or log if empty
    all_categories = all_categories_glob
    if not all_categories:
        logger.warning(f"EXPORT_HANDLER (Job {job_id}): Predefined category list (all_categories) is EMPTY. CSV columns and overall scores may be affected.")
    else:
        logger.info(f"EXPORT_HANDLER (Job {job_id}): Using predefined categories for CSV: {all_categories}")

    output = io.StringIO()
    # Define base header
    header = ['channel_url', 'channel_name', 'channel_handle', 'total_videos_analyzed', 'overall_score']
    # Add dynamic category headers
    header.extend(all_categories)
    header.extend(['controversy_status', 'controversy_reason']) # Add controversy fields
    
    writer = csv.DictWriter(output, fieldnames=header)
    writer.writeheader()

    processed_urls_count = 0
    for url, result in job_data.get('results', {}).items():
        processed_urls_count += 1
        logger.info(f"EXPORT_HANDLER (Job {job_id}, URL {processed_urls_count}): Processing result for {url}")

        row = {
            'channel_url': url,
            'channel_name': result.get('channel_name', 'Unknown'),
            'channel_handle': result.get('channel_handle', 'Unknown'),
            'total_videos_analyzed': len(result.get('video_analyses', [])),
            'overall_score': 0.0 # Default
        }
        
        # Initialize all dynamic category columns to 0.0 or ""
        for cat_name in all_categories:
            row[cat_name] = 0.0
            
        # Handle controversy screening results
        # Prioritize 'controversy_check_result' from the main result structure
        controversy_info = result.get('controversy_check_result', {})

        # Default to 'not_screened' only if controversy_info is empty or status is missing
        row['controversy_status'] = controversy_info.get('status', 'not_screened') if controversy_info else 'not_screened'
        row['controversy_reason'] = controversy_info.get('reason', '')

        summary = result.get('summary', {})
        video_analyses_data = result.get('video_analyses', [])

        if not video_analyses_data:
            logger.warning(f"EXPORT_HANDLER (Job {job_id}, URL {url}): No video_analyses found. Scores will be 0.")
            # Row will keep defaults, overall_score 0.0, categories 0.0
        
        max_overall_score = 0.0

        if summary and video_analyses_data: # Summary exists and videos were analyzed
            logger.info(f"EXPORT_HANDLER (Job {job_id}, URL {url}): Summary found. Populating category scores from summary averages and recalculating overall_score from video details.")
            # Use existing summary data for category scores (average scores)
            for category_name_from_summary, data in summary.items():
                # Fuzzy match summary category name with predefined categories
                normalized_summary_cat = category_name_from_summary.lower().replace('"', '').replace("'", "").strip()
                matched_predefined_category = None
                for predefined_cat_name in all_categories:
                    normalized_predefined_cat = predefined_cat_name.lower().strip()
                    if normalized_summary_cat == normalized_predefined_cat:
                        matched_predefined_category = predefined_cat_name
                        break
                    if normalized_predefined_cat in normalized_summary_cat or \
                       normalized_summary_cat in normalized_predefined_cat:
                        if matched_predefined_category is None: # Take first fuzzy match
                            matched_predefined_category = predefined_cat_name
                
                if matched_predefined_category:
                    score = round(data.get('average_score', 0.0), 2)
                    row[matched_predefined_category] = score
                    logger.debug(f"EXPORT_HANDLER (Job {job_id}, URL {url}): Matched summary cat '{category_name_from_summary}' to predefined '{matched_predefined_category}', avg_score: {score}")
                else:
                    logger.debug(f"EXPORT_HANDLER (Job {job_id}, URL {url}): Summary cat '{category_name_from_summary}' did not match any predefined category.")

            # Recalculate overall_score from individual video scores for accuracy (max score from any video for any recognized category)
            logger.debug(f"EXPORT_HANDLER (Job {job_id}, URL {url}): Recalculating overall_score from {len(video_analyses_data)} videos.")
            for video_idx, video_analysis in enumerate(video_analyses_data):
                analysis = video_analysis.get('analysis', {})
                llm_results_for_video = analysis.get('results', {})
                video_id_log = video_analysis.get('video_id', f'video_idx_{video_idx}')

                if not llm_results_for_video:
                    logger.debug(f"EXPORT_HANDLER (Job {job_id}, URL {url}, Video {video_id_log}): No LLM results in this video_analysis.")
                    continue

                for llm_category_name, llm_category_data in llm_results_for_video.items():
                    current_llm_score = llm_category_data.get('score', 0.0)
                    if not isinstance(current_llm_score, (int, float)) or float(current_llm_score) <= 0:
                        continue

                    normalized_llm_cat = llm_category_name.lower().replace('"', '').replace("'", "").strip()
                    
                    is_recognized_category = False
                    recognized_as = ""
                    for predefined_cat_name in all_categories:
                        normalized_predefined_cat = predefined_cat_name.lower().strip()
                        if normalized_llm_cat == normalized_predefined_cat:
                            is_recognized_category = True
                            recognized_as = predefined_cat_name
                            break
                        if normalized_predefined_cat in normalized_llm_cat or \
                           normalized_llm_cat in normalized_predefined_cat:
                            is_recognized_category = True
                            recognized_as = predefined_cat_name 
                            # Don't break, allow exact match to override fuzzy if it comes later,
                            # but for overall score, any match is fine.
                            # For simplicity, first match is fine for 'is_recognized_category'.
                            break 
                    
                    if is_recognized_category:
                        max_overall_score = max(max_overall_score, float(current_llm_score))
                        logger.debug(f"EXPORT_HANDLER (Job {job_id}, URL {url}, Video {video_id_log}): LLM cat '{llm_category_name}' (score {current_llm_score}) matched predefined '{recognized_as}'. Max_overall_score now {max_overall_score}")
                    else:
                        logger.debug(f"EXPORT_HANDLER (Job {job_id}, URL {url}, Video {video_id_log}): LLM cat '{llm_category_name}' (score {current_llm_score}) did NOT match. Not contributing to overall_score.")
            
        elif video_analyses_data: # No summary, but video analyses exist. Calculate all scores from scratch.
            logger.info(f"EXPORT_HANDLER (Job {job_id}, URL {url}): No summary found. Calculating all scores from {len(video_analyses_data)} video details.")
            for video_idx, video_analysis in enumerate(video_analyses_data):
                analysis = video_analysis.get('analysis', {})
                llm_results_for_video = analysis.get('results', {})
                video_id_log = video_analysis.get('video_id', f'video_idx_{video_idx}')

                if not llm_results_for_video:
                    logger.debug(f"EXPORT_HANDLER (Job {job_id}, URL {url}, Video {video_id_log}): No LLM results.")
                    continue

                for llm_category_name, llm_category_data in llm_results_for_video.items():
                    current_llm_score = llm_category_data.get('score', 0.0)
                    if not isinstance(current_llm_score, (int, float)) or float(current_llm_score) <= 0:
                        continue # Skip non-positive or invalid scores

                    normalized_llm_cat = llm_category_name.lower().replace('"', '').replace("'", "").strip()
                    
                    matched_predefined_category_for_this_llm_cat = None
                    best_match_is_exact = False

                    for predefined_cat_name in all_categories:
                        normalized_predefined_cat = predefined_cat_name.lower().strip()
                        if normalized_llm_cat == normalized_predefined_cat:
                            matched_predefined_category_for_this_llm_cat = predefined_cat_name
                            best_match_is_exact = True
                            break # Exact match is best
                        if not best_match_is_exact and (normalized_predefined_cat in normalized_llm_cat or \
                                                        normalized_llm_cat in normalized_predefined_cat):
                            if matched_predefined_category_for_this_llm_cat is None: # Take first fuzzy match
                                matched_predefined_category_for_this_llm_cat = predefined_cat_name
                    
                    if matched_predefined_category_for_this_llm_cat:
                        logger.debug(f"EXPORT_HANDLER (Job {job_id}, URL {url}, Video {video_id_log}): LLM cat '{llm_category_name}' (score {current_llm_score}) matched predefined '{matched_predefined_category_for_this_llm_cat}'.")
                        
                        # Update category column in CSV (max score for this category from any video so far for this channel)
                        current_category_score_in_row = row.get(matched_predefined_category_for_this_llm_cat, 0.0)
                        if float(current_llm_score) > current_category_score_in_row:
                            row[matched_predefined_category_for_this_llm_cat] = round(float(current_llm_score), 2)
                        
                        # Update overall max score
                        max_overall_score = max(max_overall_score, float(current_llm_score))
                        logger.debug(f"EXPORT_HANDLER (Job {job_id}, URL {url}, Video {video_id_log}): Max_overall_score now {max_overall_score}. Category '{matched_predefined_category_for_this_llm_cat}' row score now {row[matched_predefined_category_for_this_llm_cat]}.")
                    else:
                        logger.debug(f"EXPORT_HANDLER (Job {job_id}, URL {url}, Video {video_id_log}): LLM cat '{llm_category_name}' (score {current_llm_score}) did NOT match any predefined cat.")
        else:
             logger.info(f"EXPORT_HANDLER (Job {job_id}, URL {url}): No summary and no video_analyses. Scores remain 0.")


        row['overall_score'] = round(max_overall_score, 2)
        logger.info(f"EXPORT_HANDLER (Job {job_id}, URL {url}): Final overall_score for row: {row['overall_score']}. Category scores: {{ {', '.join([f'{k}:{v}' for k,v in row.items() if k in all_categories])} }}")
        writer.writerow(row)

    output.seek(0)
    
    # Create a proper Response object for CSV download
    response = Response(
        content=output.getvalue(), 
        media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="bulk_analysis_{job_id}.csv"'}
    )
    output.close() # Close the StringIO object
    return response

async def download_bulk_analysis_evidence(job_id: str, analysis_results: dict):
    """Download detailed evidence and transcripts as JSON"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    if job['status'] == 'processing':
        raise HTTPException(status_code=400, detail="Analysis still in progress")
    
    # Build structured evidence data
    evidence_data = {
        "job_id": job_id,
        "analysis_date": job.get('started_at'),
        "completed_date": job.get('completed_at'),
        "total_creators": len(job['results']),
        "creators": []
    }
    
    for creator_url, result in job['results'].items():
        creator_data = {
            "creator_url": creator_url,
            "channel_name": result.get('channel_name', ''),
            "channel_handle": result.get('channel_handle', ''),
            "channel_id": result.get('channel_id', ''),
            "videos": []
        }
        
        for video_analysis in result.get('video_analyses', []):
            video_data = {
                "video_url": video_analysis.get('video_url', ''),
                "video_title": video_analysis.get('video_title', ''),
                "video_id": video_analysis.get('video_id', ''),
                "transcript": "Transcript not available",  # Will be populated below
                "violations": []
            }
            
            # Get transcript from video analysis entry first (new format), then fallback to original videos data
            video_id = video_analysis.get('video_id', '')
            transcript_text = None
            
            # First try: Get transcript directly from video analysis entry (fixed format)
            if video_analysis.get('transcript'):
                transcript_text = video_analysis['transcript']
                logger.debug(f"📄 Found transcript in video analysis entry for {video_id}")
            else:
                # Fallback: Get transcript from original videos data (legacy format)
                for original_video in result.get('original_videos', []):
                    if original_video.get('id') == video_id and original_video.get('transcript'):
                        transcript_text = original_video['transcript']['full_text']
                        logger.debug(f"📄 Found transcript in original_videos for {video_id}")
                        break
            
            if transcript_text:
                video_data["transcript"] = transcript_text
            else:
                video_data["transcript"] = "Transcript not available"
                logger.warning(f"⚠️ No transcript found for video {video_id}")
            
            # Get analysis data
            analysis = video_analysis.get('analysis', {})
            
            # Add violations with evidence
            for category, violation_data in analysis.get('results', {}).items():
                violation = {
                    "category": category,
                    "score": violation_data.get('score', 0),
                    "justification": violation_data.get('justification', ''),
                    "evidence": violation_data.get('evidence', [])
                }
                video_data["violations"].append(violation)
            
            creator_data["videos"].append(video_data)
        
        evidence_data["creators"].append(creator_data)
    
    # Return as streaming JSON response
    json_str = json.dumps(evidence_data, indent=2, ensure_ascii=False)
    
    # Create output directory if it doesn't exist
    output_dir = get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON to file in output directory
    filename = f"evidence_{job_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    return FileResponse(
        filepath,
        media_type='application/json',
        filename=filename
    )

async def download_failed_urls_csv(job_id: str, analysis_results: dict):
    """Download failed URLs from bulk analysis as CSV"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    
    # Allow CSV download for any completed job (including cancelled)
    if job['status'] == 'processing':
        raise HTTPException(status_code=400, detail="Analysis still in progress")
    
    # Check if we have any failed URLs
    failed_urls = job.get('failed_urls', [])
    if not failed_urls:
        raise HTTPException(status_code=404, detail="No failed URLs to download")
    
    # Convert failed URLs to DataFrame
    rows = []
    for failed_item in failed_urls:
        row = {
            'url': failed_item.get('url', ''),
            'error_type': failed_item.get('error_type', 'unknown'),
            'error_message': failed_item.get('error', ''),
            'channel_name': failed_item.get('channel_name', ''),
            'video_id': failed_item.get('video_id', ''),
            'video_count': failed_item.get('video_count', '')
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Create output directory if it doesn't exist
    output_dir = get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    
    # Add metadata for cancelled jobs
    filename_suffix = "_partial" if job['status'] == 'cancelled' else ""
    filename = f"failed_urls_{job_id}{filename_suffix}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    return FileResponse(
        filepath,
        media_type='text/csv',
        filename=filename
    ) 