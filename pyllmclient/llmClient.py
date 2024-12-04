from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Optional, Dict, Any, Iterator, List, Union
import time
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """Configurazione base per i client LLM"""
    model_id: str = Field(default="")
    api_key: Optional[str] = None
    log_level: int = Field(default=logging.INFO)
    config: Dict[str, Any] = Field(default_factory=dict)


class LLMResponseModel(BaseModel):
    """Modello Pydantic per la risposta LLM"""
    model_id: str
    text: str
    duration: Optional[float] = None


@dataclass
class LLMResponse:
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
        self._force()
        return "".join(self._chunks)

    def to_json(self) -> Dict[str, Any]:
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
    """Abstract base class for LLM clients"""

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = LLMConfig()

        self.logger = logging.getLogger(__name__)
        self.model_id = config.model_id
        self.api_key = config.api_key
        self._is_loaded = False
        self._log_level = config.log_level
        self.config = config.config

        self.set_log_level(self._log_level)

    def set_log_level(self, level: int):
        self.logger.debug(f"Setting log level to {level}")
        self._log_level = level
        self.logger.setLevel(level)

    @abstractmethod
    def load_model(self) -> None:
        """Load and initialize the model"""
        self.logger.debug(f"Loading {self.model_id}")
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        stream: bool = True,
    ) -> Iterator[str]:
        """Generate response from the model"""
        pass

    def prompt(
        self,
        prompt: str,
        stream: bool = True,
    ) -> LLMResponse:
        """Execute prompt and return structured response"""
        try:
            if not self._is_loaded:
                self.load_model()
            return LLMResponse(self, prompt, stream=True)
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
