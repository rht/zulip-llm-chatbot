#!/usr/bin/env python3
import argparse
import configparser
import csv
import os
from pathlib import Path
from typing import Any, Dict

from langchain import LLMChain, PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

import zulip


# Taken from https://github.com/f/awesome-chatgpt-prompts/blob/86fbd9ee952449fe725767d950a0e44d8b6fe079/prompts.csv
def read_awesome_chatgpt_prompts():
    prompts = {}
    with (Path(__file__).parent / "awesome-chatgpt-prompts.csv").open() as f:
        reader = csv.reader(f)
        # Skip header
        next(reader)
        for row in reader:
            key = row[0].lower().replace(" ", "_")
            prompts[key] = row[1]
    return prompts


def initialize_llm(framework: str, token: str) -> None:
    if framework == "llama.cpp":
        from langchain.llms import LlamaCpp

        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # Verbose is required to pass to the callback manager
        model_path = "./guanaco-7B.ggmlv3.q4_0.bin"
        # Make sure the model path is correct for your system!
        return LlamaCpp(
            model_path=model_path, callback_manager=callback_manager, verbose=True
        )
    elif framework == "Hugging Face Hub":
        from langchain import HuggingFaceHub

        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

        repo_id = "google/flan-t5-xxl"
        repo_id = "stabilityai/stablelm-tuned-alpha-7b"
        # Low token output
        # repo_id = "timdettmers/guanaco-33b-merged"
        # Terse but not truncated
        repo_id = "bigscience/bloom"
        # Truncated, terse
        # repo_id = "tiiuae/falcon-7b-instruct"
        # Truncated, terse
        # repo_id = "tiiuae/falcon-7b"
        # Truncated, terse, wrong
        # repo_id = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
        # repo_id = "EleutherAI/gpt-neox-20b"
        return HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 512}
        )
    else:
        if framework != "OpenAI":
            raise Exception(f"Framework {framework} not supported")
        from langchain.llms import OpenAI

        os.environ["OPENAI_API_KEY"] = token
        return OpenAI()


class LangChainZulip:
    def __init__(
        self, config_file_name: str, prompt_file_name, enable_conversational_memory: bool = False
    ) -> None:
        if prompt_file_name is not None:
            with open(prompt_file_name) as f:
                self.template = f.read()
        else:
            self.template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""
        self.awesome_chatgpt_prompts = read_awesome_chatgpt_prompts()

        config: configparser.ConfigParser = configparser.ConfigParser()
        config.read(config_file_name)
        config_dict = dict(config["langchain"])
        zulip_config = dict(config["api"])

        self.zulip_client = zulip.Client(
            email=zulip_config["email"],
            api_key=zulip_config["key"],
            site=zulip_config["site"],
        )

        framework = config_dict.get("framework", "OpenAI")
        token = config_dict["token"]
        self.bot_name = config_dict["bot_name"]

        self.llm = initialize_llm(framework, token)
        if enable_conversational_memory:
            # https://www.pinecone.io/learn/langchain-conversational-memory/
            self.conversation_memory = ConversationSummaryBufferMemory(
                llm=self.llm, max_token_limit=650
            )
            self.prompt = None
            if prompt_file_name is not None:
                self.prompt = PromptTemplate(
                    template=self.template, input_variables=["history", "input"]
                )
            self.llm_chain = ConversationChain(
                prompt=self.prompt, llm=self.llm, memory=self.conversation_memory
            )
        else:
            self.prompt = PromptTemplate(
                template=self.template, input_variables=["question"]
            )
            self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

    def process_message(self, message: str) -> None:
        at_mention = f"@**{self.bot_name}**"

        def startswith(prefix: str) -> bool:
            return message.startswith(f"{at_mention} {prefix}")

        if startswith("!show_prompt"):
            return self.template
        elif startswith("!set_prompt"):
            template = message.replace(f"{at_mention} !set_prompt", "")
            self.template = template
            self.prompt = PromptTemplate(
                template=self.template, input_variables=["question"]
            )
            return "Prompt updated!"
        elif startswith("!set_prompt_from_templates"):
            prompt_key = message.split()[2]
            return self.llm_chain.run(
                message.replace(prompt_key, self.awesome_chatgpt_proms[prompt_key])
            )
        elif startswith("!clear_memory"):
            self.llm_chain.memory.clear()
            return "Memory forgotten!"
        return self.llm_chain.run(message)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_filename",
        help="Path to the config file",
        action="store",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_filename",
        help="Path to the prompt file",
        action="store",
    )
    args = parser.parse_args()
    lcz = LangChainZulip(args.config_filename, args.prompt_filename, enable_conversational_memory=True)

    def handle_message(msg: Dict[str, Any]) -> None:
        print("processing", msg)
        if msg["type"] != "stream":
            return

        message = msg["content"]
        content = lcz.process_message(message)
        request = {
            "type": "stream",
            "to": msg["display_recipient"],
            "topic": msg["subject"],
            "content": content,
        }
        print("sending", content)
        lcz.zulip_client.send_message(request)

    def watch_messages() -> None:
        print("Watching for messages...")

        def handle_event(event: Dict[str, Any]) -> None:
            if "message" not in event:
                # ignore heartbeat events
                return
            handle_message(event["message"])

        # https://zulip.com/api/real-time-events
        narrow = [["is", "mentioned"]]
        lcz.zulip_client.call_on_each_event(
            handle_event,
            event_types=["message"],
            all_public_streams=True,
            narrow=narrow,
        )

    watch_messages()
