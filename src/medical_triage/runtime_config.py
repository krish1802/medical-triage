from __future__ import annotations

import os
from dataclasses import dataclass


class RuntimeConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeConfig:
    hf_token: str
    api_base_url: str
    model_name: str


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeConfigError(
            f"Missing required OpenEnv runtime variable: {name}. "
            "Expected HFTOKEN, APIBASEURL, and MODELNAME to be set."
        )
    return value


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        hf_token=_required_env("HFTOKEN"),
        api_base_url=_required_env("APIBASEURL"),
        model_name=_required_env("MODELNAME"),
    )