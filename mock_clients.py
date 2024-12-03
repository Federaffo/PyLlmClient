from typing import Iterator, Optional
import random
import time
from models import LLMClient
import logging

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenAIMockClient(LLMClient):
    def __init__(self):
        super().__init__()
        self.model_id = "gpt-3.5-turbo"
        self.can_stream = True
        logger.info(f"Initialized {self.__class__.__name__}")

    def load_model(self) -> None:
        self._is_loaded = True
        logger.debug("Mock OpenAI model loaded")

    def validate_key(self) -> bool:
        return super().validate_key()

    def generate(
        self,
        prompt: str,
        stream: bool = True,
    ) -> Iterator[str]:
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        
        mock_response = "This is a mock OpenAI response."
        if stream:
            for word in mock_response.split():
                time.sleep(random.uniform(0.1, 0.3))
                yield word + " "
        else:
            yield mock_response

class AnthropicMockClient(LLMClient):
    def __init__(self):
        super().__init__()
        self.model_id = "claude-2"
        self.can_stream = True
        logger.info(f"Initialized {self.__class__.__name__}")

    def load_model(self) -> None:
        self._is_loaded = True
        logger.debug("Mock Anthropic model loaded")

    def validate_key(self) -> bool:
        return super().validate_key()

    def generate(
        self,
        prompt: str,
        stream: bool = True,
    ) -> Iterator[str]:
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        
        mock_response = "This is a mock Anthropic Claude response."
        if stream:
            for word in mock_response.split():
                time.sleep(random.uniform(0.1, 0.3))
                yield word + " "
        else:
            yield mock_response 