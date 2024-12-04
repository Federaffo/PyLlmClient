import logging

from pyllmclient import AnthropicClient, OpenAIClient
import pydantic

from pyllmclient.llmClient import LLMConfig

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

config = LLMConfig(model_id="gpt-4o", api_key="sk-proj-1234567890", log_level=logging.INFO) 
client = OpenAIClient(config=config)
response = client.prompt("Your prompt here", stream=True)
for chunk in response:
    print(chunk , end='', flush=True) 

print("")

config = LLMConfig(model_id="claude-3-5-sonnet", api_key="sk-ant-1234567890", log_level=logging.INFO  )
client = AnthropicClient(config=config)
client.set_log_level(logging.DEBUG)
response = client.prompt("Your prompt here", stream=True)
for chunk in response:
    print(chunk , end='', flush=True) 

print("")


client.set_log_level(logging.ERROR)
response = client.prompt("Your prompt here", stream=False)
print(response)
