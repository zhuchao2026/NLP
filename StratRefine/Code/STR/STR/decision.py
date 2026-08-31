import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AuthenticationError
from sc2.data import Race

import utils
from llm_config import create_openai_client
from STR.local_transformers import get_transformers_model


def _stringify_entry(entry: Any) -> str:
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        if "content" in entry:
            return str(entry["content"]).strip()
        if "text" in entry:
            return str(entry["text"]).strip()
        if "command" in entry:
            return str(entry["command"]).strip()
        return json.dumps(entry, ensure_ascii=False)
    return str(entry).strip()


def load_script_entries(script_path: str | None = None, inline_text: str | None = None) -> list[str]:
    if inline_text:
        return [inline_text.strip()] if inline_text.strip() else []
    if not script_path:
        return []

    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Scripted text source not found: {path}")

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(raw_text)
        if isinstance(payload, dict) and "messages" in payload:
            payload = payload["messages"]
        if not isinstance(payload, list):
            payload = [payload]
        return [entry for item in payload if (entry := _stringify_entry(item))]

    if suffix == ".jsonl":
        payload = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        return [entry for item in payload if (entry := _stringify_entry(item))]

    blocks = [block.strip() for block in re.split(r"^\s*---+\s*$", raw_text, flags=re.MULTILINE) if block.strip()]
    if blocks:
        return blocks
    return [raw_text]


@dataclass
class PromptBundle:
    system_prompt: str
    example_input: str = ""
    example_output: str = ""

    def render_user_input(self, context: dict) -> str:
        return json.dumps(context, ensure_ascii=False)


class DecisionBackend:
    mode = "base"

    def next_command(self, context: dict, time_label: str) -> str:
        raise NotImplementedError


class SingleLLMDecisionBackend(DecisionBackend):
    mode = "single_llm"

    def __init__(self, args, prompt_bundle: PromptBundle, game_folder: str):
        self.args = args
        self.prompt_bundle = prompt_bundle
        self.game_folder = game_folder
        self.client = create_openai_client(
            api_key=getattr(args, "LLM_api_key", None),
            provider=args.LLM_provider,
            base_url=args.LLM_base_url,
        )

    def next_command(self, context: dict, time_label: str) -> str:
        user_input = self.prompt_bundle.render_user_input(context)
        utils.save_data_to_file(
            f"{time_label}\n{user_input}",
            os.path.join(self.game_folder, "input.txt"),
            add_seperator=True,
        )

        messages = [{"role": "system", "content": self.prompt_bundle.system_prompt}]
        if self.prompt_bundle.example_input and self.prompt_bundle.example_output:
            messages.append({"role": "user", "content": self.prompt_bundle.example_input})
            messages.append({"role": "assistant", "content": self.prompt_bundle.example_output})
        messages.append({"role": "user", "content": user_input})

        while True:
            try:
                output = self.client.chat.completions.create(
                    model=self.args.LLM_api_text,
                    temperature=self.args.temperature,
                    messages=messages,
                    extra_body={"enable_thinking": False},
                )
                response = (output.choices[0].message.content or "").strip()
                break
            except AuthenticationError as exc:
                raise RuntimeError(
                    f"LLM authentication failed. provider={self.args.LLM_provider}, "
                    f"base_url={self.args.LLM_base_url}, model={self.args.LLM_api_text}, error={exc}"
                ) from exc
            except Exception as exc:
                print(
                    f"LLM request failed, retrying in 7s. "
                    f"provider={self.args.LLM_provider}, base_url={self.args.LLM_base_url}, "
                    f"model={self.args.LLM_api_text}, error={exc}"
                )
                time.sleep(7)

        utils.save_data_to_file(
            f"{time_label}\n{response}",
            os.path.join(self.game_folder, "output.txt"),
            add_seperator=True,
        )
        return response


class LocalTransformersDecisionBackend(DecisionBackend):
    mode = "local_transformers"

    def __init__(self, args, prompt_bundle: PromptBundle, game_folder: str):
        self.args = args
        self.prompt_bundle = prompt_bundle
        self.game_folder = game_folder
        self.model = get_transformers_model(args.LLM_model_path)

    def next_command(self, context: dict, time_label: str) -> str:
        user_input = self.prompt_bundle.render_user_input(context)
        utils.save_data_to_file(
            f"{time_label}\n{user_input}",
            os.path.join(self.game_folder, "input.txt"),
            add_seperator=True,
        )

        messages = [{"role": "system", "content": self.prompt_bundle.system_prompt}]
        if self.prompt_bundle.example_input and self.prompt_bundle.example_output:
            messages.append({"role": "user", "content": self.prompt_bundle.example_input})
            messages.append({"role": "assistant", "content": self.prompt_bundle.example_output})
        messages.append({"role": "user", "content": user_input})

        response = self.model.chat(
            messages=messages,
            temperature=self.args.temperature,
            max_tokens=2400,
        )
        final_actions = utils.extract_final_actions(response, Race.Protoss)
        if len(final_actions) > 50:
            raise RuntimeError(f"Local Transformers response has {len(final_actions)} final actions; limit is 50.")

        utils.save_data_to_file(
            f"{time_label}\n{response}",
            os.path.join(self.game_folder, "output.txt"),
            add_seperator=True,
        )
        return response


class ScriptedTextDecisionBackend(DecisionBackend):
    mode = "scripted_text"

    def __init__(self, args, game_folder: str):
        self.args = args
        self.game_folder = game_folder
        self.cursor = 0

    def _get_entries(self) -> list[str]:
        return load_script_entries(
            script_path=getattr(self.args, "script_path", None),
            inline_text=getattr(self.args, "script_inline", None),
        )

    def next_command(self, context: dict, time_label: str) -> str:
        entries = self._get_entries()
        if not entries:
            return ""

        repeat_last = getattr(self.args, "script_repeat_last", True)
        if self.cursor >= len(entries):
            if not repeat_last:
                return ""
            index = len(entries) - 1
        else:
            index = self.cursor

        response = entries[index].strip()
        self.cursor += 1

        utils.save_data_to_file(
            f"{time_label}\n{json.dumps(context, ensure_ascii=False)}",
            os.path.join(self.game_folder, "script_context.txt"),
            add_seperator=True,
        )
        utils.save_data_to_file(
            f"{time_label}\n{response}",
            os.path.join(self.game_folder, "script_output.txt"),
            add_seperator=True,
        )
        return response


def build_decision_backend(args, prompt_bundle: PromptBundle, game_folder: str) -> DecisionBackend:
    if args.decision_mode in {"single_llm", "STRlm"}:
        if getattr(args, "LLM_provider", "").lower() == "local_transformers":
            return LocalTransformersDecisionBackend(args, prompt_bundle, game_folder)
        return SingleLLMDecisionBackend(args, prompt_bundle, game_folder)
    if args.decision_mode == "scripted_text":
        return ScriptedTextDecisionBackend(args, game_folder)
    raise ValueError(f"Unsupported decision mode: {args.decision_mode}")
