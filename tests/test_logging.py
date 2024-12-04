import pytest
import logging
from pyllmclient.llmClient import LLMResponse, LLMClient
from pyllmclient.mock_models import OpenAIClient, AnthropicClient

def test_set_log_level(caplog):
    client = OpenAIClient()
    with caplog.at_level(logging.DEBUG):
        client.set_log_level(logging.INFO)
    
    # Check that the log contains the expected message
    assert "Setting log level to 20" in caplog.text

def test_load_model_logging(caplog):
    client = OpenAIClient(api_key="test")
    client.model_id = "test_model"
    
    with caplog.at_level(logging.DEBUG):
        client.load_model()
    
    # Check that the log contains the expected message
    assert "Loading test_model" in caplog.text

def test_logging_level_change_to_error(caplog):
    client = OpenAIClient(api_key="test", model_id="test_model")
    
    # Set initial log level to INFO
    client.set_log_level(logging.ERROR)
    with caplog.at_level(logging.DEBUG):
        client.load_model()

    assert "Loading test_model" not in caplog.text
    
    client.set_log_level(logging.DEBUG)
    
    with caplog.at_level(logging.DEBUG):
        client.load_model()
        assert "Loading test_model" in caplog.text