import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Set


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _split_env_list(value: str, default: List[str]) -> List[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class APISettings:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    cors_origins: List[str] = field(
        default_factory=lambda: _split_env_list(
            os.getenv("API_CORS_ORIGINS", ""), ["http://localhost:5173"]
        )
    )

    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", "52428800"))
    allowed_upload_extensions: Set[str] = field(
        default_factory=lambda: {
            ".csv",
            ".txt",
            ".zip",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".gif",
            ".bmp",
        }
    )

    model_path: str = os.getenv("MODEL_PATH", "")
    available_models: List[str] = field(
        default_factory=lambda: _split_env_list(
            os.getenv("AVAILABLE_MODELS", ""), [os.getenv("LLM_MODEL", "gemini-3-pro")]
        )
    )
    default_model: str = os.getenv("DEFAULT_MODEL", os.getenv("LLM_MODEL", "gemini-3-pro"))
    default_memory_model: str = os.getenv(
        "MEMORY_MODEL", os.getenv("DEFAULT_MODEL", os.getenv("LLM_MODEL", "gemini-3-pro"))
    )
    default_summary_model: str = os.getenv(
        "SUMMARY_MODEL_NAME",
        os.getenv("DEFAULT_MODEL", os.getenv("LLM_MODEL", "gemini-3-pro")),
    )
    max_rollouts: int = int(os.getenv("MAX_ROLLOUTS", "5"))
    default_rollouts: int = int(os.getenv("ROLLOUT_COUNT", "3"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.6"))
    top_p: float = float(os.getenv("TOP_P", "0.95"))
    presence_penalty: float = float(os.getenv("PRESENCE_PENALTY", "1.1"))

    enable_memory_management: bool = _env_bool("ENABLE_CONTEXT_MANAGEMENT", True)
    enable_supervisor: bool = _env_bool("ENABLE_REFLECTION", True)
    enable_browser_agent: bool = _env_bool("ENABLE_BROWSER_AGENT", True)
    enable_code_executor: bool = True
    enable_debug_stream: bool = _env_bool("API_DEBUG_STREAM", False)
    full_tool_response: bool = _env_bool("API_FULL_TOOL_RESPONSE", False)
    debug_log_dir: str = os.getenv("API_DEBUG_LOG_DIR", "./logs/debug")
    fake_mode: bool = _env_bool("API_FAKE_MODE", False)
    fake_log_path: str = os.getenv(
        "API_FAKE_LOG_PATH",
        "inference/output/deepseek-ai/DeepSeek-V3.2/gaia/20260115_194046_output.log",
    )
    fake_delay_ms: int = int(os.getenv("API_FAKE_DELAY_MS", "150"))
    fake_rounds: int = int(os.getenv("API_FAKE_ROUNDS", "4"))

    @property
    def features(self) -> dict:
        return {
            "memory_management": self.enable_memory_management,
            "supervisor": self.enable_supervisor,
            "browser_agent": self.enable_browser_agent,
            "code_executor": self.enable_code_executor,
        }


@lru_cache
def get_settings() -> APISettings:
    return APISettings()


settings = get_settings()


def reload_settings() -> APISettings:
    get_settings.cache_clear()
    global settings
    settings = get_settings()
    return settings
