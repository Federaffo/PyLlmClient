import logging

from pyllmclient import AnthropicClient, OpenAIClient

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

client = OpenAIClient(model_id="gpt-4o", api_key="sk-proj-1234567890", log_level=logging.INFO)
response = client.prompt("Your prompt here", stream=True)
for chunk in response:
    print(chunk , end='', flush=True) 

print("")

client = AnthropicClient(model_id="claude-3-5-sonnet-20240620", api_key="sk-ant-1234567890", log_level=logging.INFO  )
client.set_log_level(logging.DEBUG)
response = client.prompt("Your prompt here", stream=True)
for chunk in response:
    print(chunk , end='', flush=True) 

print("")


client.set_log_level(logging.ERROR)
response = client.prompt("Your prompt here", stream=False)
print(response)
