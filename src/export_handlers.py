"""
Export handlers for CSV and evidence downloads
"""
import os
import json
import pandas as pd
from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import logging

logger = logging.getLogger(__name__)

async def download_bulk_analysis_csv(job_id: str, analysis_results: dict):
    """Download bulk analysis results as CSV"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    
    # Allow CSV download for cancelled jobs (partial results)
    if job['status'] == 'processing':
        raise HTTPException(status_code=400, detail="Analysis still in progress")
        
    # Get all categories - fix the path to be relative to project root
    src_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(src_dir)
    categories_file = os.path.join(project_root, "data", "YouTube_Controversy_Categories.csv")
    categories_df = pd.read_csv(categories_file)
    all_categories = categories_df['Category'].tolist()
    
    # Check if we have any results
    if not job['results']:
        raise HTTPException(status_code=404, detail="No results available for download")
        
    # Convert results to DataFrame
    rows = []
    for url, result in job['results'].items():
        row = {
            'url': url,
            'channel_id': result.get('channel_id', ''),
            'channel_name': result.get('channel_name', ''),
            'channel_handle': result.get('channel_handle', ''),
            'overall_score': 0.0  # Will be calculated
        }
        
        # Initialize all categories with 0.0
        for category in all_categories:
            row[category] = 0.0
        
        # Try to get scores from summary first (for completed jobs)
        summary = result.get('summary', {})
        if summary:
            # Use existing summary data
            for category, data in summary.items():
                if category in all_categories:  # Only include valid categories
                    score = round(data.get('average_score', 0), 2)
                    row[category] = score
                    row['overall_score'] = max(row['overall_score'], score)
        else:
            # No summary available (likely cancelled job) - calculate from video analyses
            logger.info(f"📊 No summary found for {url}, calculating scores from video analyses")
            
            # Initialize category score collections
            category_scores = {}
            for category in all_categories:
                category_scores[category] = []
            
            # Collect scores from all video analyses
            for video_analysis in result.get('video_analyses', []):
                analysis = video_analysis.get('analysis', {})
                for category, violation_data in analysis.get('results', {}).items():
                    if category in category_scores:
                        score = violation_data.get('score', 0)
                        if score > 0:
                            category_scores[category].append(score)
            
            # Calculate average scores for each category
            for category in all_categories:
                if category_scores[category]:
                    avg_score = sum(category_scores[category]) / len(category_scores[category])
                    row[category] = round(avg_score, 2)
                    row['overall_score'] = max(row['overall_score'], avg_score)
                else:
                    row[category] = 0.0
            
        # Round overall score
        row['overall_score'] = round(row['overall_score'], 2)
        
        # Determine status based on threshold
        if row['overall_score'] >= 0.8:
            row['status'] = 'FAIL'
        else:
            row['status'] = 'PASS'
            
        rows.append(row)
    
    # Add failed URLs as ERROR status
    for failed_url in job.get('failed_urls', []):
        row = {
            'url': failed_url.get('url', ''),
            'channel_id': failed_url.get('channel_id', ''),
            'channel_name': failed_url.get('channel_name', ''),
            'channel_handle': '',
            'overall_score': 0.0,
            'status': 'ERROR'
        }
        
        # Initialize all categories with 0.0
        for category in all_categories:
            row[category] = 0.0
            
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Reorder columns to put status first after basic info
    column_order = ['url', 'channel_id', 'channel_name', 'channel_handle', 'status', 'overall_score'] + all_categories
    df = df[column_order]
    
    # Add metadata for cancelled jobs
    filename_suffix = "_partial" if job['status'] == 'cancelled' else ""
    filename = f"bulk_analysis_{job_id}{filename_suffix}.csv"
    df.to_csv(filename, index=False)
    
    return FileResponse(
        filename,
        media_type='text/csv',
        filename=filename
    )

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
            
            # Get transcript from original videos data
            video_id = video_analysis.get('video_id', '')
            for original_video in result.get('original_videos', []):
                if original_video.get('id') == video_id and original_video.get('transcript'):
                    video_data["transcript"] = original_video['transcript']['full_text']
                    break
            
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
    
    def generate():
        yield json_str
    
    return StreamingResponse(
        generate(), 
        media_type='application/json',
        headers={
            "Content-Disposition": f"attachment; filename=evidence_{job_id}.json"
        }
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
    
    # Add metadata for cancelled jobs
    filename_suffix = "_partial" if job['status'] == 'cancelled' else ""
    filename = f"failed_urls_{job_id}{filename_suffix}.csv"
    df.to_csv(filename, index=False)
    
    return FileResponse(
        filename,
        media_type='text/csv',
        filename=filename
    ) 