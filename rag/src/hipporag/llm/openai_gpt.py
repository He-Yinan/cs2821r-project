import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from openai import AzureOpenAI
from packaging import version
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper

def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)  
        # Use exponential backoff: start with 2 seconds, max 60 seconds, multiply by 2 each time
        dynamic_retry = retry(
            stop=stop_after_attempt(max_retries), 
            wait=wait_exponential(multiplier=2, min=2, max=60),
            retry=retry_if_exception_type((openai.APIConnectionError, httpx.ConnectError, httpx.TimeoutException))
        )
        decorated_func = dynamic_retry(func)
        try:
            return decorated_func(self, *args, **kwargs)
        except (openai.APIConnectionError, httpx.ConnectError) as e:
            logger.error(f"Failed to connect to LLM server at {getattr(self, 'llm_base_url', 'unknown')} after {max_retries} retries. "
                        f"Error: {str(e)}. Please check if the server is running and accessible.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during LLM inference: {str(e)}")
            raise
    return wrapper

class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        config_dict['max_retries'] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None,
                 high_throughput: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            # Set reasonable timeouts: 30s connect, 5min read
            client = httpx.Client(
                limits=limits, 
                timeout=httpx.Timeout(30.0, connect=30.0, read=5*60, write=30.0)
            )
        else:
            client = None

        self.max_retries = kwargs.get("max_retries", 2)

        if self.global_config.azure_endpoint is None:
            self.openai_client = OpenAI(base_url=self.llm_base_url, http_client=client, max_retries=self.max_retries, api_key="EMPTY")  # MINE
        else:
            self.openai_client = AzureOpenAI(api_version=self.global_config.azure_endpoint.split('api-version=')[1],
                                             azure_endpoint=self.global_config.azure_endpoint, max_retries=self.max_retries)

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['llm_base_url'] = self.global_config.llm_base_url
        config_dict['generate_params'] = {
                "model": self.global_config.llm_name,
                "max_completion_tokens": config_dict.get("max_new_tokens", 400),
                "n": config_dict.get("num_gen_choices", 1),
                "seed": config_dict.get("seed", 0),
                "temperature": config_dict.get("temperature", 0.0),
            }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    def check_connection(self) -> bool:
        """
        Check if the LLM server is reachable by making a simple test request.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            from urllib.parse import urlparse
            import socket
            
            # Extract host and port from base_url
            parsed_url = urlparse(self.llm_base_url)
            host = parsed_url.hostname
            
            if not host:
                logger.warning(f"Invalid URL format: {self.llm_base_url}")
                return False
            
            # Determine port
            if parsed_url.port:
                port = parsed_url.port
            elif parsed_url.scheme == 'https':
                port = 443
            elif parsed_url.scheme == 'http':
                port = 80
            else:
                # Default to 80 for unknown schemes
                port = 80
            
            # Try to connect
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)  # 5 second timeout
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.info(f"Connection check successful: {self.llm_base_url} is reachable")
                return True
            else:
                logger.warning(f"Connection check failed: Cannot connect to {self.llm_base_url} (host: {host}, port: {port})")
                return False
        except socket.gaierror as e:
            logger.warning(f"DNS resolution failed for {self.llm_base_url}: {str(e)}")
            return False
        except Exception as e:
            logger.warning(f"Connection check error: {str(e)}")
            return False

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"): # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params['max_tokens'] = params.pop('max_completion_tokens')

        response = self.openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        # Handle case where content might be None (some models return None for empty responses)
        if response_message is None:
            logger.warning(f"LLM returned None content. Finish reason: {response.choices[0].finish_reason}, Completion tokens: {response.usage.completion_tokens if hasattr(response, 'usage') else 'N/A'}")
            response_message = ""
        assert isinstance(response_message, str), f"response_message should be a string, got {type(response_message)}"
        
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_message, metadata


