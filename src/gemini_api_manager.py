"""
Gemini API Key Rotation Manager

Manages multiple Gemini API keys with automatic rotation when rate limits are hit.
Provides a unified interface for all Gemini API calls with built-in error handling.
"""

import os
import time
import logging
from typing import List, Optional, Any, Dict
from google import genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DailyQuotaExhausted(Exception):
    """Raised when the daily API quota is exhausted for all keys."""
    pass

class GeminiApiManager:
    """
    Manages multiple Gemini API keys with automatic rotation on rate limits.
    
    Features:
    - Circular queue of API keys
    - Automatic rotation on rate limit errors
    - Cooldown period when all keys are rate limited
    - Centralized error handling and logging
    """
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.rate_limited_keys = set()  # Track which keys are currently rate limited
        self.daily_quota_exhausted_keys = set()  # Track which keys have daily quota exhausted
        self.last_rotation_time = 0
        self.cooldown_period = 70  # seconds (60s rate limit window + 10s buffer)
        self.cooldown_cycles = 0  # Track number of full cooldown cycles
        self.max_cooldown_cycles = 10  # Maximum cooldown cycles before giving up
        
        if not self.api_keys:
            raise ValueError("No Gemini API keys found. Please set GOOGLE_API_KEY_1 through GOOGLE_API_KEY_5")
        
        logger.info(f"🔑 Initialized Gemini API Manager with {len(self.api_keys)} keys")
        
        # Initialize the first client
        self._current_client = None
        self._initialize_current_client()
    
    def _load_api_keys(self) -> List[str]:
        """Load API keys from environment variables."""
        keys = []
        for i in range(1, 6):  # GOOGLE_API_KEY_1 through GOOGLE_API_KEY_5
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if key:
                keys.append(key)
        
        # Fallback to original GOOGLE_API_KEY if no numbered keys found
        if not keys:
            original_key = os.getenv("GOOGLE_API_KEY")
            if original_key:
                keys.append(original_key)
                logger.warning("⚠️ Using single GOOGLE_API_KEY. For rotation, set GOOGLE_API_KEY_1 through GOOGLE_API_KEY_5")
        
        return keys
    
    def _initialize_current_client(self):
        """Initialize the genai client with the current API key."""
        if self.api_keys:
            current_key = self.api_keys[self.current_key_index]
            self._current_client = genai.Client(api_key=current_key)
            logger.info(f"🔄 Initialized client with key #{self.current_key_index + 1}")
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if the error indicates a rate limit."""
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit",
            "quota exceeded",
            "too many requests",
            "429",
            "resource exhausted",
            "requests per minute",
            "rpm"
        ]
        
        # Check for HTTP 429 status code
        if hasattr(error, 'status_code') and error.status_code == 429:
            return True
        
        # Check error message for rate limit indicators
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def _extract_retry_delay(self, error: Exception) -> Optional[float]:
        """
        Extract retry delay from error message if available.
        Returns delay in seconds, or None if not found.
        """
        import re
        error_str = str(error)
        
        try:
            # Try to find "Please retry in X.XXs" pattern
            match = re.search(r'retry in (\d+\.?\d*)s', error_str)
            if match:
                delay = float(match.group(1))
                logger.info(f"📊 API suggests retry delay: {delay:.1f}s")
                return delay
            
            # Try to find retryDelay in JSON
            if 'retryDelay' in error_str:
                match = re.search(r"'retryDelay':\s*'(\d+)s'", error_str)
                if match:
                    delay = float(match.group(1))
                    logger.info(f"📊 API suggests retry delay: {delay:.1f}s")
                    return delay
        except Exception as parse_error:
            logger.warning(f"⚠️ Failed to parse retry delay: {parse_error}")
        
        return None
    
    def _is_daily_quota_error(self, error: Exception) -> bool:
        """
        Check if the error indicates a daily quota exhaustion.
        Looks for 'PerDay' patterns in quotaId field.
        """
        error_str = str(error)
        
        # Look for daily quota indicators
        daily_indicators = [
            'PerDay',
            'PerDayPerUser',
            'PerProjectPerDay',
            'RequestsPerDay'
        ]
        
        # Check if any daily indicator is in the error message
        if any(indicator in error_str for indicator in daily_indicators):
            logger.error(f"🚫 Daily quota exhausted detected in error")
            return True
        
        return False
    
    def _rotate_to_next_key(self, is_daily_quota: bool = False) -> bool:
        """
        Rotate to the next available API key.
        
        Args:
            is_daily_quota: Whether the current key hit daily quota (vs temporary RPM limit)
        
        Returns:
            bool: True if rotation successful, False if all keys are rate limited
        """
        if len(self.api_keys) == 1:
            logger.warning("⚠️ Only one API key available - cannot rotate")
            return False
        
        # Mark current key as rate limited
        self.rate_limited_keys.add(self.current_key_index)
        
        # If this is a daily quota error, mark it separately
        if is_daily_quota:
            self.daily_quota_exhausted_keys.add(self.current_key_index)
            logger.warning(f"🚫 Key #{self.current_key_index + 1} hit DAILY quota limit")
        else:
            logger.warning(f"🚫 Key #{self.current_key_index + 1} hit rate limit")
        
        # Check if ALL keys have daily quota exhausted
        if len(self.daily_quota_exhausted_keys) == len(self.api_keys):
            logger.error("❌ ALL API keys have daily quota exhausted")
            return False
        
        # Try to find a non-rate-limited key (prioritize keys without daily quota exhaustion)
        original_index = self.current_key_index
        attempts = 0
        
        while attempts < len(self.api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            attempts += 1
            
            # Skip keys with daily quota exhausted
            if self.current_key_index in self.daily_quota_exhausted_keys:
                continue
            
            if self.current_key_index not in self.rate_limited_keys:
                self._initialize_current_client()
                self.last_rotation_time = time.time()
                logger.info(f"🔄 Rotated to key #{self.current_key_index + 1}")
                return True
        
        # All non-daily-exhausted keys are rate limited (temporary)
        logger.warning(f"⚠️ All available API keys are temporarily rate limited")
        return False
    
    def _wait_for_cooldown(self, suggested_delay: Optional[float] = None):
        """Wait for cooldown period and reset temporarily rate limited keys."""
        # Increment cooldown cycle counter
        self.cooldown_cycles += 1
        
        # Check if we've exceeded max cooldown cycles
        if self.cooldown_cycles > self.max_cooldown_cycles:
            logger.error(f"❌ Exceeded maximum cooldown cycles ({self.max_cooldown_cycles}). Giving up.")
            raise DailyQuotaExhausted(
                f"Exceeded {self.max_cooldown_cycles} cooldown cycles. "
                "This likely indicates a daily quota exhaustion. "
                "Please check your API quotas or add more API keys."
            )
        
        # Always use suggested delay if available, with +10s buffer as requested
        if suggested_delay:
            wait_time = suggested_delay + 10  # Add 10s buffer to suggested delay
            logger.info(f"⏳ All keys rate limited. Using API suggested delay + buffer: {wait_time:.1f}s")
        else:
            wait_time = self.cooldown_period  # Fall back to 70s
            logger.info(f"⏳ All keys rate limited. Waiting default cooldown: {wait_time:.1f}s")
        
        logger.info(f"📊 Cooldown cycle {self.cooldown_cycles}/{self.max_cooldown_cycles}")
        time.sleep(wait_time)
        
        # Reset ONLY temporarily rate limited keys (not daily quota exhausted ones)
        self.rate_limited_keys.clear()
        
        # Find first key that doesn't have daily quota exhausted
        for i in range(len(self.api_keys)):
            if i not in self.daily_quota_exhausted_keys:
                self.current_key_index = i
                break
        
        self._initialize_current_client()
        logger.info(f"🔄 Cooldown complete. Reset to key #{self.current_key_index + 1}")
    
    def _make_api_call(self, api_call_func, max_retries: int = None) -> Any:
        """
        Make an API call with automatic key rotation on rate limits.
        
        Args:
            api_call_func: Function that makes the API call using self._current_client
            max_retries: Maximum number of key rotations to try (defaults to number of keys * 4)
            
        Returns:
            API response
            
        Raises:
            DailyQuotaExhausted: If daily quota is exhausted
            Exception: If all retries exhausted or non-rate-limit error occurs
        """
        if max_retries is None:
            max_retries = len(self.api_keys) * 4  # Allow multiple cooldown cycles
        
        last_error = None
        suggested_delay = None
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                # Make the API call
                result = api_call_func(self._current_client)
                # Reset cooldown cycles on success
                self.cooldown_cycles = 0
                return result
                
            except Exception as e:
                last_error = e
                
                if self._is_rate_limit_error(e):
                    logger.warning(f"🚫 Rate limit hit on attempt {attempt + 1}")
                    
                    # Check if this is a daily quota error for the current key
                    is_daily_quota = self._is_daily_quota_error(e)
                    
                    # Extract suggested retry delay from error
                    delay = self._extract_retry_delay(e)
                    if delay:
                        suggested_delay = delay
                    
                    # Try to rotate to next key (pass whether this was daily quota)
                    rotation_result = self._rotate_to_next_key(is_daily_quota=is_daily_quota)
                    
                    # If rotation failed because ALL keys have daily quota exhausted
                    if not rotation_result and len(self.daily_quota_exhausted_keys) == len(self.api_keys):
                        logger.error("❌ Daily quota exhausted for ALL API keys")
                        raise DailyQuotaExhausted(
                            "Daily API quota exhausted for all keys. Please wait for quota reset or add more API keys. "
                            f"Error: {str(e)[:200]}"
                        )
                    
                    if rotation_result:
                        continue  # Try again with new key
                    else:
                        # All available keys temporarily rate limited - wait for cooldown
                        try:
                            self._wait_for_cooldown(suggested_delay)
                            suggested_delay = None  # Reset for next cycle
                            continue  # Try again after cooldown
                        except DailyQuotaExhausted:
                            # Re-raise daily quota exhaustion
                            raise
                else:
                    # Non-rate-limit error - don't rotate, just re-raise
                    logger.error(f"❌ Non-rate-limit error in Gemini API call: {e}")
                    raise e
        
        # If we get here, we've exhausted all retries
        logger.error(f"❌ All {max_retries} retry attempts failed. Last error: {last_error}")
        raise last_error
    
    @property
    def client(self) -> genai.Client:
        """Get the current Gemini client (for backward compatibility)."""
        return self._current_client
    
    def generate_content(self, model: str, contents: str, config: Optional[Dict] = None, **kwargs) -> Any:
        """
        Generate content using Gemini with automatic key rotation.
        
        Args:
            model: Model name (e.g., "gemini-2.5-flash")
            contents: Input content/prompt
            config: Generation config (response_mime_type, response_schema, etc.)
            **kwargs: Additional arguments
            
        Returns:
            Generation response
        """
        def api_call(client):
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config or {},
                **kwargs
            )
        
        return self._make_api_call(api_call)
    
    def embed_content(self, model: str, contents: List[str], task_type: str = None, output_dimensionality: int = None, **kwargs) -> Any:
        """
        Generate embeddings using Gemini with automatic key rotation.
        
        Args:
            model: Embedding model name (e.g., "gemini-embedding-001")
            contents: List of texts to embed
            task_type: Task type for embedding ("RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY" or "CLUSTERING")
            output_dimensionality: Output dimension for embeddings (e.g., 768)
            **kwargs: Additional arguments
            
        Returns:
            Embedding response
        """
        def api_call(client):
            # Build config if task_type or output_dimensionality specified
            config = None
            if task_type or output_dimensionality:
                from google.genai import types
                config = types.EmbedContentConfig()
                if task_type:
                    config.task_type = task_type
                if output_dimensionality:
                    config.output_dimensionality = output_dimensionality
            
            return client.models.embed_content(
                model=model,
                contents=contents,
                config=config,
                **kwargs
            )
        
        return self._make_api_call(api_call)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the API manager."""
        return {
            "total_keys": len(self.api_keys),
            "current_key_index": self.current_key_index,
            "rate_limited_keys": list(self.rate_limited_keys),
            "daily_quota_exhausted_keys": list(self.daily_quota_exhausted_keys),
            "available_keys": [i for i in range(len(self.api_keys)) if i not in self.rate_limited_keys and i not in self.daily_quota_exhausted_keys]
        }


# Global instance for backward compatibility
_gemini_manager = None

def get_gemini_manager() -> GeminiApiManager:
    """Get the global Gemini API manager instance."""
    global _gemini_manager
    if _gemini_manager is None:
        _gemini_manager = GeminiApiManager()
    return _gemini_manager

def get_gemini_client() -> genai.Client:
    """Get the current Gemini client (backward compatibility function)."""
    return get_gemini_manager().client

def get_current_api_key() -> str:
    """Get the current active API key for external libraries that need it directly."""
    manager = get_gemini_manager()
    return manager.api_keys[manager.current_key_index]
