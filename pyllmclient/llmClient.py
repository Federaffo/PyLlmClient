from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Optional, Dict, Any, Iterator, List, Union
import time
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """Base configuration for LLM clients"""
    model_id: str = Field(default="")
    api_key: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class LLMResponseModel(BaseModel):
    """Modello Pydantic per la risposta LLM"""
    model_id: str
    text: str
    duration: Optional[float] = None


@dataclass
class LLMResponse:
    """Response wrapper for LLM generations.
    
    This class handles both streaming and non-streaming responses from LLM models,
    collecting chunks and providing access to the complete response.
    
    Attributes:
        model (LLMClient): The LLM client instance that generated the response
        prompt (str): The input prompt used to generate the response
        stream (bool): Whether the response is streamed or not
    """
    model: 'LLMClient'
    prompt: str
    stream: bool
    _done: bool = False
    _chunks: List[str] = None
    _start: Optional[float] = None
    _end: Optional[float] = None

    def __init__(self, model: 'LLMClient', prompt: str, stream: bool = True):
        self.model = model
        self.prompt = prompt
        self.stream = stream
        self._done = False
        self._chunks = []
        self._start = None
        self._end = None

    def __str__(self) -> str:
        return self.text()

    def _force(self):
        if not self._done:
            list(self)

    def text(self) -> str:
        """Returns the complete generated text.
        
        Forces completion if the generation is not finished.
        
        Returns:
            str: The complete generated text
        """
        self._force()
        return "".join(self._chunks)

    def to_json(self) -> Dict[str, Any]:
        """Converts the response to a JSON-serializable dictionary.
        
        Returns:
            Dict[str, Any]: Response data including model ID, text, and duration
        """
        self._force()
        duration = None
        if self._start and self._end:
            duration = self._end - self._start
        
        response = LLMResponseModel(
            model_id=self.model.model_id,
            text=self.text(),
            duration=duration
        )
        return response.model_dump()

    def __iter__(self) -> Iterator[str]:
        self._on_start()
        if self._done:
            yield from self._chunks
            return

        try:
            self.model.logger.info(f"Prompting {self.model.model_id}")
            for chunk in self.model.generate(
                self.prompt,
                stream=self.stream,
            ):
                yield chunk
                self._chunks.append(chunk)
        except Exception as e:
            self.model.handle_error(e)

        self._on_done()

    def _on_start(self):
        self._start = time.monotonic()

    def _on_done(self):
        self._end = time.monotonic()
        self._done = True


class LLMClient(ABC):
    """Abstract base class for LLM clients.
    
    This class provides the basic interface and functionality for implementing
    specific LLM model clients.
    
    Attributes:
        model_id (str): Identifier for the LLM model
        api_key (Optional[str]): API key for authentication
        config (Dict[str, Any]): Additional configuration parameters
        logger_level (int): Logging level for the client
    """

    def __init__(self, config: Optional[LLMConfig] = None , log_level: int = logging.INFO):
        if config is None:
            config = LLMConfig()

        self.logger = logging.getLogger(__name__)
        self.model_id = config.model_id
        self.api_key = config.api_key
        self._is_loaded = False
        self._log_level = log_level
        self.config = config.config

        self.set_log_level(self._log_level)

    def set_log_level(self, level: int):
        self.logger.debug(f"Setting log level to {level}")
        self._log_level = level
        self.logger.setLevel(level)

    @abstractmethod
    def load_model(self) -> None:
        """Load and initialize the model"""
        self.logger.info(f"Loading {self.model_id}")
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        stream: bool,
    ) -> Iterator[str]:
        """Generate response from the model"""
        pass

    def prompt(
        self,
        prompt: str,
        stream: bool,
    ) -> LLMResponse:
        """Execute a prompt and return a structured response.
        
        Args:
            prompt (str): The input text to send to the model
            stream (bool): Whether to stream the response or not
            
        Returns:
            LLMResponse: A response object containing the generated text
            
        Raises:
            Exception: Any errors during model loading or generation
        """
        try:
            if not self._is_loaded:
                self.load_model()
            return LLMResponse(self, prompt, stream=stream)
        except Exception as e:
            self.handle_error(e)

    def handle_error(self, error: Exception):
        """
        Default error handling implementation.
        This method can be overridden
            by subclasses to provide custom error handling.
        """
        self.logger.error(f"Default error handling: {error}")
        raise error

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.model_id}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.model_id}'>"
