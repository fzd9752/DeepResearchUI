import json
import json5
import os
import re
from pathlib import Path
import traceback
import time
import asyncio
from typing import Dict, List, Optional, Union
from base_tool import (
    Message,
    BaseTool,
)
from openai import OpenAI
from transformers import AutoTokenizer
import datetime as dt
from datetime import datetime

try:
    from pydemo.common.polaris_helper import PolarisHelper
except ImportError:
    PolarisHelper = None

from prompt import *
from supervisor import Supervisor
from context.tools import create_context_manage_tool

from tool_search import *
from tool_visit import *
from tool_code import *

# ========== 全局配置 ==========
# 这里集中读取所有依赖的环境变量，方便统一管理和排查。
MAX_LLM_CALL_PER_RUN = int(os.getenv("MAX_LLM_CALL_PER_RUN", 75))
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3-pro")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")
AGENT_API_BASE = os.getenv("AGENT_API_BASE", "")
ENABLE_CONTEXT_MANAGEMENT = os.getenv(
    "ENABLE_CONTEXT_MANAGEMENT", "0"
).strip().lower() in ("1", "true", "yes", "y", "on")

TOOL_CLASS = [
    Scholar(),
    Visit(),
    Search(),
    CodeExecutor(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import random


def today_date():
    """返回当前日期（YYYY-MM-DD），让系统提示词具备当天语境。"""
    return dt.date.today().strftime("%Y-%m-%d")


class Agent:
    """多轮 ReAct Agent，负责串起 LLM、工具调用和上下文管理。"""

    # 配置常量
    TIMEOUT_MINUTES = 90
    MAX_TOKENS = 110 * 1024
    MCP_NAMESPACE = "Development"
    MCP_SERVICE = "trpc.aimate.aiground_mcp_server.httpProxy"
    MCP_TIMEOUT = 5
    MEMORY_RETRY_NUM = 1

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
        llm: Optional[Union[Dict]] = None,
        event_callback=None,
        cancel_checker=None,
        enable_context_management: Optional[bool] = None,
        enable_supervisor: Optional[bool] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        self.function_list = function_list or []
        self.llm = llm
        self.kwargs = kwargs
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.browser_traces: List[dict] = []
        self.current_rollout_idx: Optional[int] = None
        self.event_callback = event_callback
        self.cancel_checker = cancel_checker
        self.enable_context_management = (
            ENABLE_CONTEXT_MANAGEMENT
            if enable_context_management is None
            else enable_context_management
        )
        self.emit_full_tool_response = bool(kwargs.get("emit_full_tool_response", False))
        self.model_name = model_name
        self.supervisor = Supervisor(
            enable_reflection=enable_supervisor,
            llm_caller=None,
            event_callback=self.event_callback,
        )
        self._tokenizer = None
        self.context_mgr = create_context_manage_tool()
        self.supervisor.llm_caller = self.call_server

    def sanity_check_output(self, content):
        """快速检查输出是否包含 <think> 标签，用于最基本的格式校验。"""
        return "<think>" in content and "</think>" in content

    def emit_event(self, event_type: str, data: dict) -> None:
        if self.event_callback:
            self.event_callback(event_type, data)

    def call_server(self, msgs, max_tries=10):
        """对 OpenAI 兼容 API 接口发起请求，并带指数退避重试。"""

        openai_api_key = AGENT_API_KEY or os.getenv("OPENAI_API_KEY", "")
        openai_api_base = AGENT_API_BASE

        # Use OpenAI client
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1
        for attempt in range(max_tries):
            try:
                print(
                    f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---"
                )
                chat_response = None

                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get("temperature", 0.6),
                    top_p=self.llm_generate_cfg.get("top_p", 0.95),
                    max_tokens=10000,
                )

                # 统一处理不同模型的响应格式
                message = chat_response.choices[0].message

                # 提取推理内容（thinking/reasoning）
                reasoning_content = getattr(message, "reasoning_content", "") or ""

                # 提取主要内容（处理列表或字符串格式）
                raw_content = message.content
                if isinstance(raw_content, list):
                    content = raw_content[0].get("text", "") if raw_content else ""
                else:
                    content = raw_content or ""

                # Handle truncated responses due to length limit
                if not content and chat_response.choices[0].finish_reason == "length":
                    print(
                        f"[call_server] Warning: Response was truncated due to length limit (attempt {attempt + 1}) ======> Response: {chat_response}"
                    )
                    should_retry, updated_msgs, action_desc = (
                        self.supervisor.handle_truncated_response(
                            reasoning_content, msgs
                        )
                    )
                    print(f"[call_server] Supervisor action: {action_desc}")

                    if should_retry:
                        msgs = updated_msgs
                        if attempt < max_tries - 1:
                            sleep_time = base_sleep_time * (
                                2**attempt
                            ) + random.uniform(0, 1)
                            sleep_time = min(sleep_time, 30)
                            time.sleep(sleep_time)
                            continue
                        else:
                            print(
                                "[call server] Error: All retry attempts have been exhausted due to truncated responses. The call has failed."
                            )
                            return f"vllm server error!!!"
                    else:
                        return (
                            content
                            if content
                            else "[call_server] Empty response due to length limit."
                        )

                if reasoning_content and reasoning_content.strip():
                    content = (
                        "<think>\n"
                        + reasoning_content
                        + "\n</think>"
                        + "\n\n"
                        + content
                    )

                if content and content.strip():
                    print("--- Service call successful, received a valid response ---")
                    return content.strip()
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")

            except Exception as e:
                print(
                    f"[call server] Error: Attempt {attempt + 1} failed with an unexpected error: {e}. ======> Response: {chat_response}"
                )

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2**attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)

                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print(
                    "[call server] Error: All retry attempts have been exhausted. The call has failed."
                )

        return f"vllm server error!!!"

    def count_tokens(self, messages):
        """计算当前上下文 token 数，防止超过 LLM 上限。"""
        if not self.llm_local_path:
            if self._tokenizer is None:
                print("[count_tokens] MODEL_PATH is empty; skip token counting.")
                self._tokenizer = False
            return 0

        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
            except Exception as exc:
                print(
                    f"[count_tokens] Failed to load tokenizer from {self.llm_local_path}: {exc}"
                )
                self._tokenizer = False
                return 0

        if self._tokenizer is False:
            return 0

        full_prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = self._tokenizer(full_prompt, return_tensors="pt")
        token_count = len(tokens["input_ids"][0])

        return token_count

    def _build_result(
        self,
        question: str,
        answer: str,
        messages: List,
        traces: List,
        prediction: str,
        termination: str,
        is_used_memory: bool,
    ) -> Dict:
        """构建统一的结果结构。"""
        return {
            "question": question,
            "answer": answer,
            "messages": messages,
            "traces": traces,
            "prediction": prediction,
            "termination": termination,
            "use_context_management": self.enable_context_management,
            "is_used_memory": is_used_memory,
        }

    def _extract_question(self, data: dict) -> str:
        """从 data 中提取 question。"""
        try:
            return data["item"]["question"]
        except:
            raw_msg = data["item"]["messages"][1]["content"]
            return raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

    def _parse_tool_call(self, content: str, messages: List[Dict] = None) -> tuple:
        """解析工具调用，返回 (is_python, tool_name, tool_args_or_code, error)。"""
        match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if not match:
            return False, None, None, "Tool call parsing error: missing <tool_call> block"
        tool_call_str = match.group(1).strip()

        # 检查是否是 Python 代码执行
        if (
            "codeexecutor" in tool_call_str.lower()
            and "<code>" in tool_call_str.lower()
        ):
            try:
                code = (
                    content.split("<tool_call>")[1]
                    .split("</tool_call>")[0]
                    .split("<code>")[1]
                    .split("</code>")[0]
                    .strip()
                )
                return True, "CodeExecutor", code, None
            except Exception as e:
                error = f"[Code Executor Error]: Formatting error. {e}"
                # 尝试通过 supervisor 纠正
                if messages is not None:
                    new_content = self.supervisor.handle_tool_parsing_error(
                        error, tool_call_str, messages
                    )
                    if new_content:
                        print(f"[Agent] Retrying with corrected response")
                        # 递归调用，用新响应重新解析（不再传递 messages 避免无限递归）
                        return self._parse_tool_call(new_content, messages=None)
                return True, None, None, error

        if tool_call_str.lower().startswith("<code>"):
            try:
                code = (
                    tool_call_str.split("<code>")[1]
                    .split("</code>")[0]
                    .strip()
                )
                return True, "CodeExecutor", code, None
            except Exception as e:
                return True, None, None, f"[Code Executor Error]: Formatting error. {e}"

        # 优先解析 <name>/<arguments> 标签格式，避免不必要的纠错调用
        if "<name>" in tool_call_str and "</name>" in tool_call_str:
            name_match = re.search(r"<name>(.*?)</name>", tool_call_str, re.DOTALL)
            args_match = re.search(
                r"<arguments>(.*?)</arguments>", tool_call_str, re.DOTALL
            )
            if name_match:
                tool_name = name_match.group(1).strip()
                args_text = args_match.group(1).strip() if args_match else ""
                tool_args = self._parse_arguments_block(args_text)
                return False, tool_name, tool_args, None
        elif tool_call_str.lstrip().startswith("<"):
            # 明显是 XML 风格但未包含 <name>，避免触发纠错调用
            return False, None, None, "Tool call parsing error: unsupported tag format"

        # 解析 JSON 格式的工具调用
        try:
            json_block = tool_call_str
            if "<code>" in tool_call_str.lower():
                json_block = tool_call_str.split("<code>", 1)[0].strip()
            tool_call = json5.loads(json_block)
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})
            return False, tool_name, tool_args, None
        except Exception as e:
            error = f"Tool call parsing error: {e}"
            # 尝试通过 supervisor 纠正
            if messages is not None:
                new_content = self.supervisor.handle_tool_parsing_error(
                    error, tool_call_str, messages
                )
                if new_content:
                    print(f"[Agent] Retrying with corrected response")
                    # 递归调用，用新响应重新解析（不再传递 messages 避免无限递归）
                    return self._parse_tool_call(new_content, messages=None)
            # 回退：解析 <name>/<arguments> 标签格式
            name_match = re.search(r"<name>(.*?)</name>", tool_call_str, re.DOTALL)
            args_match = re.search(
                r"<arguments>(.*?)</arguments>", tool_call_str, re.DOTALL
            )
            if name_match:
                tool_name = name_match.group(1).strip()
                args_text = args_match.group(1).strip() if args_match else ""
                tool_args = self._parse_arguments_block(args_text)
                return False, tool_name, tool_args, None
            return False, None, None, error

    def _parse_arguments_block(self, args_text: str) -> dict:
        if not args_text:
            return {}
        try:
            return json5.loads(args_text)
        except Exception:
            pass
        args = {}
        for tag, value in re.findall(r"<(\w+)>(.*?)</\1>", args_text, re.DOTALL):
            cleaned = value.strip()
            if not cleaned:
                continue
            parsed = cleaned
            try:
                parsed = json5.loads(cleaned)
            except Exception:
                parsed = cleaned
            args[tag] = parsed
        return args if args else {"raw": args_text}

    def _extract_tag_content(self, content: str, tag: str) -> Optional[str]:
        """Extract content between tags like <tag>...</tag>."""
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        if open_tag not in content or close_tag not in content:
            return None
        return content.split(open_tag)[1].split(close_tag)[0].strip()

    def _handle_memory_update(
        self,
        question: str,
        round_index: int,
        latest_observation: str,
        latest_response: str,
    ) -> Optional[str]:
        """Update memory list with latest response and observation."""
        mem_start_time = time.time()
        for i in range(self.MEMORY_RETRY_NUM):
            try:
                latest_memory_unit = self.context_mgr.update_memory(
                    question, round_index, latest_observation, latest_response
                )
                print(f"[Memory] Memory Time Cost: {time.time() - mem_start_time} s.")
                print(f"[Memory] The Latest Memory Unit: {latest_memory_unit}")
                memory_payload = None
                try:
                    memory_payload = json.loads(latest_memory_unit)
                except Exception:
                    memory_payload = None
                if memory_payload:
                    self.emit_event(
                        "memory_update",
                        {
                            "rollout_id": self.current_rollout_idx,
                            "round": round_index,
                            "action": "update",
                            "memory_unit": memory_payload,
                        },
                    )
                return latest_memory_unit
            except Exception as e:
                if i < self.MEMORY_RETRY_NUM - 1:
                    print(
                        f"[Memory] Memory Update Warning: {e}, {traceback.format_exc()}. Retrying {i + 1}/{self.MEMORY_RETRY_NUM}..."
                    )
                    continue
                else:
                    print(
                        f"[Memory] Memory Updtae Error: {e}, {traceback.format_exc()}"
                    )
                    return

    async def find_one_address(self, namespace: str, service_name: str) -> str:
        polaris_helper = PolarisHelper()
        node = polaris_helper.find_one_node(
            namespace, service_name, env_name="test"
        )  # setname
        return node.address

    async def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        """Agent 主循环：驱动推理、解析工具调用并组装最终结果。"""
        print(f"\nData: {data}\n")
        print(f"Context Mangagement:{self.enable_context_management}")

        # ========== Agent Model ==========
        # 使用全局配置的模型名称
        self.model = self.model_name or LLM_MODEL
        self.browser_traces = []
        self.current_rollout_idx = data.get("rollout_idx")

        # ========== Prompt Construct ==========
        question = self._extract_question(data)
        answer = data["item"]["answer"]
        self.user_prompt = question
        start_time = time.time()
        system_prompt = SYSTEM_PROMPT + str(today_date())
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        traces = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN

        # ========== Agent Run ==========
        round = 0
        is_used_memory = False

        while num_llm_calls_available > 0:
            if self.cancel_checker and self.cancel_checker():
                return self._build_result(
                    question,
                    answer,
                    messages,
                    traces,
                    "Task cancelled",
                    "cancelled",
                    is_used_memory,
                )
            result = None
            latest_memory_unit = None
            # 超时保护
            elapsed_time = time.time() - start_time
            timeout_seconds = self.TIMEOUT_MINUTES * 60

            # 检查是否已经超时
            if elapsed_time > timeout_seconds:
                return self._build_result(
                    question,
                    answer,
                    messages,
                    traces,
                    "No answer found after timeout",
                    "timeout",
                    is_used_memory,
                )

            # 检查是否接近超时（90% 的时间）
            if elapsed_time > timeout_seconds * 0.9 and num_llm_calls_available > 1:
                print(
                    f"[call server] Warning: Approaching timeout limit ({elapsed_time / 60:.1f}/{self.TIMEOUT_MINUTES:.0f} min)."
                )
                should_intervene, result_dict, action_desc = (
                    self.supervisor.handle_approaching_timeout(
                        messages, elapsed_time, self.TIMEOUT_MINUTES
                    )
                )
                print(f"[Supervisor] {action_desc}")

                if should_intervene and result_dict:
                    prediction, termination = (
                        result_dict["prediction"],
                        result_dict["termination"],
                    )
                else:
                    content = self.call_server(messages)
                    messages.append({"role": "assistant", "content": content.strip()})
                    traces.append({"role": "assistant", "content": content.strip()})
                    prediction, termination = (
                        self.supervisor.format_limit_reached_result(content, "timeout")
                    )

                return self._build_result(
                    question,
                    answer,
                    messages,
                    traces,
                    prediction,
                    termination,
                    is_used_memory,
                )

            round += 1
            self.emit_event(
                "round_start",
                {
                    "rollout_id": self.current_rollout_idx,
                    "round": round,
                    "started_at": datetime.utcnow().isoformat() + "Z",
                },
            )
            num_llm_calls_available -= 1
            content = self.call_server(messages, max_tries=5)
            print(f"\n {'=' * 30} Round {round}: {'=' * 30} \n{content}")
            if "<tool_response>" in content:
                pos = content.find("<tool_response>")
                content = content[:pos]

            thinking_content = self._extract_tag_content(content, "think")
            if thinking_content:
                self.emit_event(
                    "round_thinking",
                    {
                        "rollout_id": self.current_rollout_idx,
                        "round": round,
                        "content": thinking_content,
                    },
                )

            # 检查模型输出是否包含完整的工具调用或最终答案
            has_complete_tool_call = (
                "<tool_call>" in content and "</tool_call>" in content
            )
            has_complete_answer = "<answer>" in content and "</answer>" in content

            if not (has_complete_tool_call or has_complete_answer):
                print(
                    f"[call_server] Warning: No tool call or final answer detected in the response.======> Response: {content}"
                )
                messages.append({"role": "assistant", "content": content.strip()})

                should_retry, updated_messages, action_desc = (
                    self.supervisor.handle_missing_action(messages)
                )
                print(f"[Supervisor] {action_desc}")

                if should_retry:
                    messages = updated_messages
                    content = self.call_server(messages, max_tries=5)
                    messages.append({"role": "assistant", "content": content.strip()})

            # ========== Tool ==========
            if "<tool_call>" in content and "</tool_call>" in content:
                tool_start_time = time.time()
                is_python, tool_name, tool_data, error = self._parse_tool_call(
                    content, messages
                )

                if error:
                    result = error
                    print(f"[Tool call] {error}")
                    self.emit_event(
                        "round_complete",
                        {
                            "rollout_id": self.current_rollout_idx,
                            "round": round,
                            "status": "failed",
                            "error": error,
                            "duration_ms": int((time.time() - tool_start_time) * 1000),
                        },
                    )
                elif is_python:
                    self.emit_event(
                        "round_acting",
                        {
                            "rollout_id": self.current_rollout_idx,
                            "round": round,
                            "tool": tool_name,
                            "arguments": {},
                            "code": tool_data,
                        },
                    )
                    result = TOOL_MAP["CodeExecutor"].call(tool_data)
                else:
                    self.emit_event(
                        "round_acting",
                        {
                            "rollout_id": self.current_rollout_idx,
                            "round": round,
                            "tool": tool_name,
                            "arguments": tool_data,
                        },
                    )
                    if tool_name == "visit":
                        tool_data["question"] = question
                    result = self.custom_call_tool(tool_name, tool_data)

                raw_result = result
                result = "<tool_response>\n" + result + "\n</tool_response>"
                print(f"\nTool Response:\n{result}\n")

                if not error:
                    preview = raw_result.strip().replace("\n", " ")
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    observing_payload = {
                        "rollout_id": self.current_rollout_idx,
                        "round": round,
                        "tool": tool_name,
                        "result_summary": f"Tool returned {len(raw_result)} chars",
                        "result_preview": preview,
                    }
                    if self.emit_full_tool_response:
                        observing_payload["result_full"] = raw_result
                    self.emit_event("round_observing", observing_payload)
                    self.emit_event(
                        "round_complete",
                        {
                            "rollout_id": self.current_rollout_idx,
                            "round": round,
                            "status": "success",
                            "tool": tool_name,
                            "duration_ms": int(
                                (time.time() - tool_start_time) * 1000
                            ),
                        },
                    )

                # ========== Memory List Update ==========
                latest_memory_unit = None
                if self.enable_context_management:
                    latest_memory_unit = self._handle_memory_update(
                        question, round, result, content
                    )
                    latest_memory_unit = (
                        json.loads(latest_memory_unit) if latest_memory_unit else None
                    )

            llm_calls_used = MAX_LLM_CALL_PER_RUN - num_llm_calls_available
            overall_progress = (
                llm_calls_used / MAX_LLM_CALL_PER_RUN if MAX_LLM_CALL_PER_RUN else 0
            )
            progress_value = float(f"{overall_progress:.4f}")
            elapsed_seconds = time.time() - start_time
            estimated_remaining_seconds = None
            if overall_progress > 0:
                estimated_remaining_seconds = (
                    elapsed_seconds * (1 - overall_progress) / overall_progress
                )

            self.emit_event(
                "progress_update",
                {
                    "overall_progress": progress_value,
                    "elapsed_seconds": int(elapsed_seconds),
                    "estimated_remaining_seconds": (
                        int(estimated_remaining_seconds)
                        if estimated_remaining_seconds is not None
                        else None
                    ),
                    "llm_calls": {
                        "current": llm_calls_used,
                        "max": MAX_LLM_CALL_PER_RUN,
                    },
                    "rollouts_status": [
                        {
                            "id": self.current_rollout_idx,
                            "progress": progress_value,
                            "rounds": round,
                        }
                    ],
                },
            )

            # ========== Answer ==========
            if "<answer>" in content and "</answer>" in content:
                messages.append({"role": "assistant", "content": content.strip()})
                traces.append({"role": "assistant", "content": content.strip()})
                # 检测到模型产出最终答案时立即结束循环
                termination = "answer"
                break

            # ========== Context Manage ==========
            # Only process if there was a tool call (result is set)
            if result is None:
                messages.append({"role": "assistant", "content": content.strip()})
                traces.append({"role": "assistant", "content": content.strip()})
                continue

            # First Round / Updated Memory Failed/ Context Manage Unable/ Continue Current sub-goal
            if (
                round == 1
                or not latest_memory_unit
                or len(latest_memory_unit["rounds_index"]) > 1
            ):
                # messages append
                messages.append({"role": "assistant", "content": content.strip()})
                messages.append({"role": "user", "content": result})
            # Start new sub-goal
            else:
                # Get memory list
                memory_list = self.context_mgr.get_memory()
                # Retain system prompt & user question
                last_messages_num = len(messages) + 2
                last_messages_tokens = self.count_tokens(messages) + self.count_tokens(
                    [
                        {"role": "assistant", "content": content.strip()},
                        {"role": "user", "content": result},
                    ]
                )
                messages = messages[:2]
                # Replace the message of the completed sub-goal with memory
                completed_sub_goal_messages = json.loads(memory_list)[:-1]
                completed_sub_goal_messages = json.dumps(
                    completed_sub_goal_messages, ensure_ascii=False
                )
                messages.append(
                    {"role": "user", "content": completed_sub_goal_messages}
                )
                messages.append({"role": "assistant", "content": content.strip()})
                messages.append({"role": "user", "content": result})
                print(
                    f"============ [Context Management] After Context Management ============\n"
                    f"- Messages Num: {last_messages_num} -> {len(messages)}\n"
                    f"- Messages Tokens: {last_messages_tokens} -> {self.count_tokens(messages)}\n"
                    f"- Used Memory Content: {completed_sub_goal_messages}\n"
                )
                traces.append({"role": "user", "content": completed_sub_goal_messages})
                is_used_memory = True
            traces.append({"role": "assistant", "content": content.strip()})
            traces.append({"role": "user", "content": result})

            if num_llm_calls_available <= 0 and "<answer>" not in content:
                messages[-1]["content"] = (
                    "Sorry, the number of llm calls exceeds the limit."
                )

            if num_llm_calls_available <= 1:
                print(
                    f"[call server] Warning: Approaching the maximum number of LLM calls allowed."
                )
                should_intervene, result_dict, action_desc = (
                    self.supervisor.handle_approaching_limit(
                        messages, num_llm_calls_available
                    )
                )
                print(f"[Supervisor] {action_desc}")

                if should_intervene and result_dict:
                    prediction, termination = (
                        result_dict["prediction"],
                        result_dict["termination"],
                    )
                else:
                    content = self.call_server(messages)
                    messages.append({"role": "assistant", "content": content.strip()})
                    traces.append({"role": "assistant", "content": content.strip()})
                    prediction, termination = (
                        self.supervisor.format_limit_reached_result(content, "llm_call")
                    )

                return self._build_result(
                    question,
                    answer,
                    messages,
                    traces,
                    prediction,
                    termination,
                    is_used_memory,
                )

            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > self.MAX_TOKENS:
                print(
                    f"[call server] Token quantity exceeds the limit: {token_count} > {self.MAX_TOKENS}"
                )

                should_intervene, result_dict, action_desc = (
                    self.supervisor.handle_token_limit_exceeded(
                        messages, token_count, self.MAX_TOKENS
                    )
                )
                print(f"[Supervisor] {action_desc}")

                if should_intervene and result_dict:
                    prediction, termination = (
                        result_dict["prediction"],
                        result_dict["termination"],
                    )
                else:
                    content = self.call_server(messages)
                    messages.append({"role": "assistant", "content": content.strip()})
                    traces.append({"role": "assistant", "content": content.strip()})
                    prediction, termination = (
                        self.supervisor.format_limit_reached_result(content, "token")
                    )

                self._persist_browser_traces(data)
                return self._build_result(
                    question,
                    answer,
                    messages,
                    traces,
                    prediction,
                    termination,
                    is_used_memory,
                )

        # 解析最终答案
        if "<answer>" in messages[-1]["content"]:
            prediction = (
                messages[-1]["content"].split("<answer>")[1].split("</answer>")[0]
            )
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = (
                "exceed available llm calls"
                if num_llm_calls_available == 0
                else "answer not found"
            )

        self._persist_browser_traces(data)
        return self._build_result(
            question, answer, messages, traces, prediction, termination, is_used_memory
        )

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        """统一入口，根据工具名将请求分发到实际实现，并做必要的参数兼容。"""
        if tool_name in TOOL_MAP:
            # 某些旧工具依赖 params 字段，这里统一补充
            # 保存原始参数的副本，避免后续修改影响 params
            import copy

            original_args = copy.deepcopy(tool_args)

            # 准备工具调用所需的完整参数
            tool_args["params"] = original_args

            if tool_name == "visit":
                final_question = getattr(self, "user_prompt", "")
                sub_goal = tool_args.get("goal", "")
                combined_goal_parts = []
                if final_question:
                    combined_goal_parts.append(f"Final question: {final_question}")
                if sub_goal:
                    combined_goal_parts.append(f"Current sub question: {sub_goal}")
                if combined_goal_parts:
                    tool_args["goal"] = "\n".join(combined_goal_parts)
            if "python" in tool_name.lower() or "code" in tool_name.lower():
                # 代码执行工具需要保持向后兼容的参数命名
                result = TOOL_MAP["CodeExecutor"].call(tool_args)
            else:
                # import pdb; pdb.set_trace()
                # 所有参数都已整合到 tool_args 中
                raw_result = TOOL_MAP[tool_name].call(tool_args)
                result = raw_result
                if tool_name == "visit":
                    self._try_capture_visit_trace(tool_args)
            return result

        else:
            return f"Error: Tool {tool_name} not found"

    def _try_capture_visit_trace(self, tool_args: dict):
        """从 visit 工具抓取最新的浏览器轨迹并缓存，供后续持久化。"""
        visit_tool = TOOL_MAP.get("visit")
        trace_getter = getattr(visit_tool, "get_last_trace", None)
        if not trace_getter:
            return
        trace = trace_getter()
        if not trace:
            return
        try:
            safe_args = json.loads(json.dumps(tool_args))
        except Exception:
            safe_args = str(tool_args)
        record = {
            "question": getattr(self, "user_prompt", ""),
            "rollout_idx": getattr(self, "current_rollout_idx", None),
            "tool_args": safe_args,
            "trace": trace,
            "timestamp": time.time(),
        }
        self.browser_traces.append(record)

    def _resolve_trace_dir(self, data: dict) -> Optional[str]:
        """推断轨迹输出目录，优先使用显式环境变量。"""
        env_dir = os.getenv("BROWSER_TRACE_DIR")
        if env_dir:
            return env_dir
        output_base = os.getenv("OUTPUT_PATH") or os.getenv("OUTPUT_DIR") or ""
        dataset_path = (
            os.getenv("DATASET")
            or os.getenv("DATA_PATH")
            or data.get("dataset_path", "")
        )
        model_path = self.llm_local_path or os.getenv("MODEL_PATH") or ""
        if not output_base or not dataset_path:
            return None
        model_name = os.path.basename(model_path.rstrip("/")) if model_path else "model"
        dataset_name = (
            os.path.basename(dataset_path.rstrip("/"))
            .replace(".jsonl", "")
            .replace(".json", "")
        )
        return os.path.join(output_base, f"{model_name}_sglang", dataset_name)

    def _build_trace_filename(self, question: str, rollout_idx: Optional[int]) -> str:
        """轨迹统一落单文件，追加写入。"""
        return "browser_traces.jsonl"

    def _persist_browser_traces(self, data: dict):
        """将缓存的浏览器轨迹落盘，统一写入单个文件（追加模式）。"""
        if not self.browser_traces:
            return
        trace_dir = self._resolve_trace_dir(data)
        if not trace_dir:
            print("[browser trace] Skip persist: trace dir unresolved")
            return
        try:
            Path(trace_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[browser trace] mkdir failed: {exc}")
            return
        filename = self._build_trace_filename(
            getattr(self, "user_prompt", "") or "question",
            getattr(self, "current_rollout_idx", None),
        )
        path = Path(trace_dir) / filename
        try:
            with path.open("a", encoding="utf-8") as f:
                for record in self.browser_traces:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[browser trace] write failed: {exc}")
