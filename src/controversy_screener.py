"""
Controversy screening functionality for YouTube creators
"""
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

async def screen_creator_for_controversy(channel_name: str, channel_handle: str, llm_analyzer, channel_url: str = None) -> Tuple[bool, str, str]:
    """
    Pre-screen a creator for ongoing controversies using LLM.
    Returns (is_controversial, reason, status)
    where status is one of: 'controversial', 'not_controversial', 'error'
    """
    try:
        logger.info(f"🔍 CONTROVERSY CHECK: Screening creator - Name: '{channel_name}', Handle: '{channel_handle}', URL: '{channel_url}'")
        
        # If we don't have a proper channel name, try to extract it from the URL or handle
        effective_name = channel_name
        if not channel_name or channel_name in ['Unknown', 'unknown', '']:
            if channel_handle and channel_handle not in ['Unknown', 'unknown', '']:
                effective_name = channel_handle.replace('@', '')
                logger.info(f"🔍 Using handle as name: {effective_name}")
            elif channel_url:
                # Try to extract a name from the URL
                if '/@' in channel_url:
                    effective_name = channel_url.split('/@')[1].split('/')[0]
                    logger.info(f"🔍 Extracted name from URL handle: {effective_name}")
                elif '/c/' in channel_url:
                    effective_name = channel_url.split('/c/')[1].split('/')[0]
                    logger.info(f"🔍 Extracted name from URL custom: {effective_name}")
                elif '/user/' in channel_url:
                    effective_name = channel_url.split('/user/')[1].split('/')[0]
                    logger.info(f"🔍 Extracted name from URL user: {effective_name}")
        
        # If we still don't have a good name, we can't do effective screening
        if not effective_name or effective_name in ['Unknown', 'unknown', '']:
            logger.warning(f"🔍 CONTROVERSY CHECK: No effective name found for screening - URL: {channel_url}")
            return False, "Unable to determine creator identity for controversy screening", "error"
        
        # Create a prompt to check for ongoing controversies
        prompt = f"""
        You are a content moderation assistant. Your task is to determine if a YouTube creator is currently involved in any significant ongoing controversies.
        
        Creator Information:
        - Channel Name: {effective_name}
        - Channel Handle: {channel_handle if channel_handle and channel_handle not in ['Unknown', 'unknown'] else 'Not available'}
        - YouTube URL: {channel_url if channel_url else 'Not available'}
        
        Check if this creator is currently involved in any of the following:
        1. Major public scandals or controversies that are CURRENTLY ACTIVE (within the last 6 months)
        2. Active legal issues, criminal investigations, or ongoing court cases
        3. Recent serious allegations of misconduct that are still being investigated or discussed
        4. Significant community backlash for harmful behavior that is ONGOING
        5. Recent content that has led to platform strikes, demonetization, or channel warnings
        
        Important guidelines:
        - Only flag creators with CURRENT, ONGOING, and SIGNIFICANT controversies
        - Historical issues that have been resolved should NOT be flagged
        - Minor disagreements or typical internet drama should NOT be flagged
        - Educational content about controversial topics should NOT be flagged
        - Political opinions or religious views should NOT be flagged unless they involve hate speech
        - Be specific about the timeframe - controversies must be recent and ongoing
        
        Known controversial creators to flag (if matched):
        - Creators currently facing criminal charges
        - Creators with active investigations for serious misconduct
        - Creators who have been banned from major platforms in the last 6 months
        - Creators involved in ongoing legal disputes about harmful content
        
        Respond with a JSON object:
        {{
            "is_controversial": true/false,
            "reason": "Brief explanation if controversial, or 'No ongoing controversies found' if not",
            "confidence": "high/medium/low"
        }}
        
        Be conservative - only flag if you are confident there is a significant ongoing controversy.
        """
        
        logger.debug(f"�� CONTROVERSY CHECK: Sending prompt to LLM for {effective_name}")
        
        # Use the LLM to check
        response = await llm_analyzer.check_controversy_async(prompt)
        
        logger.info(f"🔍 CONTROVERSY CHECK: LLM Response for {effective_name}: {response}")
        
        if response and isinstance(response, dict):
            # Only flag if confidence is high or medium
            confidence = response.get('confidence', 'low')
            if confidence == 'low':
                logger.info(f"🔍 CONTROVERSY CHECK: Low confidence for {effective_name}, not flagging")
                return False, "Low confidence in controversy assessment", "not_controversial"
            
            is_controversial = response.get('is_controversial', False)
            reason = response.get('reason', 'Unknown')
            
            # Log the decision for debugging
            if is_controversial:
                logger.info(f"🚫 Controversy check: {effective_name} flagged with {confidence} confidence - {reason}")
                return True, reason, "controversial"
            else:
                logger.info(f"✅ Controversy check: {effective_name} passed screening with {confidence} confidence")
                return False, reason, "not_controversial"
        
        # Default to not controversial if check fails
        logger.warning(f"Controversy check returned invalid response for {effective_name}, defaulting to not controversial")
        return False, "Controversy check returned invalid response", "error"
        
    except Exception as e:
        logger.error(f"Failed to screen creator {channel_name} for controversies: {str(e)}", exc_info=True)
        # On error, log but don't block the creator
        return False, f"Controversy screening check failed: {str(e)}", "error" 