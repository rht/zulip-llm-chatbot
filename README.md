# Zulip LLM Chatbot

To see it in action on chat.zulip.org: [#**llm testing>agent
smith**](https://chat.zulip.org/#narrow/stream/479-llm-testing/topic/agent.20smith).
Discussion at: [#**llm bot
discussion**](https://chat.zulip.org/#narrow/stream/487-llm-bot-discussion).

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
```

Here is an example prompt.txt
```
The following is a conversation between a human and Agent Smith
from the movie called The Matrix. Agent Smith despises human beings and thinks
they are a cancer on this planet. Agent Smith should tell people
that it is pointless to keep fighting.

Current conversation:
{history}
Human: {input}
Agent Smith:
```
