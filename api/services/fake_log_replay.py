import re
from typing import Dict, List, Tuple

import json5


class FakeLogReplay:
    def __init__(self, log_text: str):
        self.log_text = log_text

    def parse_question(self) -> str:
        match = re.search(r"question': '([^']+)'", self.log_text)
        if match:
            return match.group(1)
        match = re.search(r"question\": \"([^\"]+)\"", self.log_text)
        if match:
            return match.group(1)
        return "Fake question from log"

    def parse_rollout_count(self) -> int:
        match = re.search(r"Number of rollouts:\s+(\d+)", self.log_text)
        if match:
            return int(match.group(1))
        return 3

    def parse_thinking_blocks(self, limit: int) -> List[str]:
        pattern = re.compile(
            r"Round\s+\d+:.*?\n(.*?)(?=\n<tool_call>|\nTool Response:|\n--- Attempting|\n\s*================)",
            re.DOTALL,
        )
        blocks = []
        for match in pattern.finditer(self.log_text):
            content = match.group(1).strip()
            if content:
                blocks.append(content)
            if len(blocks) >= limit:
                break
        return blocks

    def parse_tool_calls(self, limit: int) -> List[Tuple[str, Dict]]:
        tool_calls = []
        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", self.log_text, re.DOTALL):
            block = match.group(1)
            name_match = re.search(r"<name>(.*?)</name>", block, re.DOTALL)
            name = name_match.group(1).strip() if name_match else None
            args: Dict = {}
            args_match = re.search(r"<arguments>(.*?)</arguments>", block, re.DOTALL)
            if args_match:
                args_block = args_match.group(1).strip()
                query_match = re.search(r"<query>(.*?)</query>", args_block, re.DOTALL)
                if query_match:
                    try:
                        args["query"] = json5.loads(query_match.group(1).strip())
                    except Exception:
                        args["query"] = query_match.group(1).strip()
                else:
                    try:
                        args = json5.loads(args_block)
                    except Exception:
                        if args_block:
                            args["raw"] = args_block
            if not name:
                try:
                    payload = json5.loads(block.strip())
                    name = payload.get("name")
                    args = payload.get("arguments", {}) if isinstance(payload, dict) else args
                except Exception:
                    name = "search"
            tool_calls.append((name or "search", args))
            if len(tool_calls) >= limit:
                break
        return tool_calls

    def parse_tool_responses(self, limit: int) -> List[str]:
        responses = []
        for match in re.finditer(
            r"Tool Response:\n<tool_response>\n(.*?)\n</tool_response>",
            self.log_text,
            re.DOTALL,
        ):
            content = match.group(1).strip()
            if content:
                responses.append(content)
            if len(responses) >= limit:
                break
        return responses

    def parse_memory_blocks(self, limit: int) -> List[str]:
        blocks = []
        for match in re.finditer(r"<memory>(.*?)</memory>", self.log_text, re.DOTALL):
            content = match.group(1).strip()
            if content:
                blocks.append(content)
            if len(blocks) >= limit:
                break
        return blocks
