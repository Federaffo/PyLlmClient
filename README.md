# PyLlmClient

## Overview

PyLlmClient is a Python library designed to interact with language models, providing a streamlined interface for generating text responses. It supports multiple backend models, including OpenAI and Anthropic, and offers features such as streaming responses and error handling.

## Installation

To install the required dependencies, run:

bash
pip install -r requirements.txt



## Usage

### Basic Example

Here's a basic example of how to use the PyLlmClient with OpenAI and Anthropic clients:

python
import logging
from src.mock_models import AnthropicClient, OpenAIClient

Initialize OpenAI client
client = OpenAIClient(model_id="gpt-4o", api_key="your-api-key", log_level=logging.INFO)
response = client.prompt("Your prompt here", stream=True)
for chunk in response:
print(chunk, end='', flush=True)

Initialize Anthropic client
client = AnthropicClient(model_id="claude-3-5-sonnet-20240620", api_key="your-api-key", log_level=logging.INFO)
client.set_log_level(logging.DEBUG)
response = client.prompt("Your prompt here", stream=True)
for chunk in response:
print(chunk, end='', flush=True)


### Running Tests

To run the tests, you can use the following command:


bash
pytest tests



This will execute the test suite located in the `tests` directory, ensuring that all components of the library are functioning as expected.

## Project Structure

- `src/`: Contains the main source code for the library.
  - `llmClient.py`: Defines the abstract base class `LLMClient` and the `LLMResponse` class.
  - `mock_models.py`: Implements the `OpenAIClient` and `AnthropicClient` classes.
- `tests/`: Contains unit tests for the library.
- `examples/`: Provides example scripts demonstrating how to use the library.

## Development

### Code Style

The project uses `flake8` for checking PEP8 style compliance. You can run the style check with:
