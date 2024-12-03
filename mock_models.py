import logging
import time
from typing import Iterator, Optional
from models import LLMClient


class OpenAIClient(LLMClient):
    def __init__(self, model_id: str = "gpt-4", api_key: Optional[str] = None, log_level: int = logging.DEBUG):
        super().__init__(log_level = log_level)
        self.model_id = model_id
        self.api_key = api_key
        self.can_stream = True

    def load_model(self) -> None:
        if self.api_key is None:
            raise ValueError("API key is required") 
        # Initialize OpenAI client
        logging.debug(f"Loading {self.model_id}")
        self._is_loaded = True

    def generate(
        self,
        prompt: str,
        stream: bool = True,
    ) -> Iterator[str]:
        # Implement OpenAI specific generation
        response = "this is a very long response"
        if stream:
            for i in response.split():
                time.sleep(0.2)  # Simulate response delay
                yield f"{i} "
        else:
            yield response

class AnthropicClient(LLMClient):
    def __init__(self, model_id: str = "claude-3-5-sonnet-20240620", api_key: Optional[str] = None, log_level: int = logging.INFO):
        super().__init__(log_level)
        self.model_id = model_id
        self.api_key = api_key
        self.can_stream = True

    def load_model(self) -> None:
        if self.api_key is None:
            raise ValueError("API key is required") 
        # Initialize Anthropic client
        logging.debug(f"Loading {self.model_id}")
        self._is_loaded = True

    def generate(
        self,
        prompt: str,
        stream: bool = True,
    ) -> Iterator[str]:
        # Implement Anthropic specific generation
        response = "this is a very long response from anthropic" 
        if stream:
            for i in response.split():
                time.sleep(0.1)  # anthropic is faster :)
                yield f"{i} "
        else:
            yield response