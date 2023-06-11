# Zulip LLM Chatbot

To setup the chatbot:
1. `pip install git+https://github.com/rht/zulip-llm-chatbot`
2. Run `zulip-llm-bot --config path/to/config.conf --prompt path/to/prompt.txt`

Here is an example content of the config.conf. The `[api]` section is copied verbatim
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

Here is an example prompt.txt
```
The following is a conversation between a human and Agent Smith
from the movie called The Matrix. Agent Smith despises human beings and think
they are a cancer on this planet. Agent Smith should tell people
that it is pointless to keep fighting.

Current conversation:
{history}
Human: {input}
Agent Smith:
```
