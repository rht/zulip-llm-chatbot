# Zulip LLM Chatbot

To setup the chatbot:
1. `pip install git+https://github.com/rht/zulip-llm-chatbot`
2. Run `zulip-llm-bot --config path/to/config.conf --prompt path/to/prompt.txt`

An example content of the config.conf. The `[api]` section is copied verbatim
from the zuliprc.
```
[api]
email = llm-bot@chat.zulip.org
key = Specify your Zulip key here
site=https://chat.zulip.org

[langchain]
framework = OpenAI
token = <Specify your token here>
bot_name = <LLMBot -- bot name in your Zulip organization>
```
