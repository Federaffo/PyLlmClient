import logging
import time
from typing import Iterator, Optional
from pyllmclient import LLMClient
from pyllmclient.llmClient import LLMConfig 


class OpenAIClient(LLMClient):
    def __init__(self, 
                 model_id: str = "gpt-4",
                 api_key: Optional[str] = None,
                 config: Optional[LLMConfig] = None,
                 log_level: int = logging.DEBUG):
        if config is None:
            config = LLMConfig(
                model_id=model_id,
                api_key=api_key,
                log_level=log_level
            )
        super().__init__(config=config)
        self.can_stream = True

    def load_model(self) -> None:
        if self.api_key is None:
            raise ValueError("API key is required")
        # Initialize OpenAI client
        self.logger.debug(f"Loading {self.model_id}")
        self._is_loaded = True

    def generate(
        self,
        prompt: str,
        stream: bool = True,
    ) -> Iterator[str]:
        # Implement OpenAI specific generation
        response = "This is a very long response"
        if stream:
            for i in response.split():
                time.sleep(0.2)  # Simulate response delay
                yield f"{i} "
        else:
            yield response


class AnthropicClient(LLMClient):
    def __init__(self,
                 model_id: str = "claude-3-5-sonnet-20240620",
                 api_key: Optional[str] = None,
                 config: Optional[LLMConfig] = None,
                 log_level: int = logging.INFO):
        if config is None:
            config = LLMConfig(
                model_id=model_id,
                api_key=api_key,
                log_level=log_level
            )
        super().__init__(config=config)
        self.can_stream = True

    def load_model(self) -> None:
        if self.api_key is None:
            raise ValueError("API key is required")
        # Initialize Anthropic client
        self.logger.debug(f"Loading {self.model_id}")
        self._is_loaded = True

    def generate(
        self,
        prompt: str,
        stream: bool = True,
    ) -> Iterator[str]:
        # Implement Anthropic specific generation
        response = "This is a very long response from anthropic"
        if stream:
            for i in response.split():
                time.sleep(0.1)  # anthropic is faster :)
                yield f"{i} "
        else:
            yield response
