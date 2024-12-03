import pytest
from src.llmClient import LLMResponse, LLMClient
from src.mock_models import OpenAIClient, AnthropicClient

# Tutti i metodi con questo decorator sono stati generati da AI (e talvolta modificati)
#def ai_generated(func):
#    return
#    def wrapper(*args, **kwargs):
#        return func(*args, **kwargs)
#    return wrapper

@pytest.fixture
def openai_client():
    return OpenAIClient(model_id="gpt-4", api_key="test")

@pytest.fixture
def anthropic_client():
    return AnthropicClient(model_id="claude-3-5-sonnet" , api_key="test")

def test_llmresponse_initialization(openai_client):
    response = LLMResponse(openai_client, "Test prompt")
    assert response.model == openai_client
    assert response.prompt == "Test prompt"
    assert response.stream is True
    assert response._done is False
    assert response._chunks == []

def test_llmresponse_text(openai_client):
    response = LLMResponse(openai_client, "Test prompt")
    response._chunks = ["Hello", " ", "world"]
    response._done = True
    assert response.text() == "Hello world"

def test_llmresponse_to_json(openai_client):
    response = LLMResponse(openai_client, "Test prompt")
    response._chunks = ["Hello", " ", "world"]
    response._done = True
    expected_json = {
        "model_id": openai_client.model_id,
        "text": "Hello world"
    }
    assert response.to_json() == expected_json

def test_llmresponse_iteration(openai_client):
    response = LLMResponse(openai_client, "Test prompt")
    response._chunks = ["Hello", " ", "world"]
    response._done = True
    assert list(response) == ["Hello", " ", "world"]

def test_llmclient_load_model(openai_client):
    openai_client.load_model()
    assert openai_client._is_loaded is True

def test_llmclient_prompt(openai_client):
    response = openai_client.prompt("Test prompt")
    assert isinstance(response, LLMResponse)

def test_llmclient_handle_error(openai_client):
    with pytest.raises(Exception):
        openai_client.handle_error(Exception("Test error"))

def test_openai_mock_generate(openai_client):
    response = list(openai_client.generate("Test prompt"))
    assert response == ["This ", "is ", "a ", "very ", "long ", "response "]

def test_anthropic_mock_generate(anthropic_client):
    response = list(anthropic_client.generate("Test prompt"))
    assert response == ["This ", "is ","a ", "very ", "long ", "response ", "from ", "anthropic "]

def test_openai_client_without_api_key():
    with pytest.raises(ValueError, match="API key is required"):
        model = OpenAIClient(model_id="gpt-4")
        model.load_model()

def test_anthropic_client_without_api_key():
    with pytest.raises(ValueError, match="API key is required"):
        model = AnthropicClient(model_id="claude-3-5-sonnet")
        model.load_model()
