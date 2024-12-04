
# PyLLMClient

PyLLMClient is a Python library designed to simplify interactions with Large Language Models (LLMs). 
It provides a standardized interface, response handling, and error management for building integrations with LLMs.

## Features

- **Abstract Base Class**: Easily extendable `LLMClient` for custom implementations.
- **Response Handling**: Collects streaming and non-streaming responses with the `LLMResponse` wrapper.
- **Configuration Management**: Use `LLMConfig` for simple and structured configurations.
- **Pydantic Integration**: Leverage Pydantic models for validation and serialization of responses.

## Requirements

- Python 3.8 or higher
- Dependencies listed in `pyproject.toml`

## Installation

To install the library, clone the repository and use `pip`:

```bash
pip install .
```

## Usage

### Basic Example

```python
from PyLLMClient.llmclient import LLMClient, LLMConfig

class MyCustomLLM(LLMClient):
    def load_model(self):
        self.logger.info(f"Custom LLM {self.model_id} loaded.")
        self._is_loaded = True

    def generate(self, prompt: str, stream: bool = True):
        self.logger.info(f"Generating response for: {prompt}")
        yield "Mock response from MyCustomLLM"

# Configuration
config = LLMConfig(model_id="custom-llm", api_key="your_api_key")

# Initialize and use the LLM
client = MyCustomLLM(config=config)
response = client.prompt("Hello, world!", stream=False)
print(response.text())
```

## Testing

Run unit tests using `pytest`:

```bash
pytest
```

Or use Docker for testing:

### Build and Run Docker Tests

1. Build the test image:

```bash
docker build -t pyllmclient-tests .
```

2. Run the tests:

```bash
docker run pyllmclient-tests
```

3. Alternatively, use docker-compose:

```bash
docker-compose run tests
```

## Contributing

Contributions are welcome! Please fork the repository, make changes, and submit a pull request.

## License

This library is licensed under the MIT License. See the `LICENSE` file for details.
