from pathlib import Path
from typing import Any, Dict, TypeVar, Union, cast

import tomli
import zero


RawConfig = Dict[str, Any]
Report = Dict[str, Any]
T = TypeVar('T')


class Timer(zero.Timer):
    @classmethod
    def launch(cls) -> 'Timer':
        timer = cls()
        timer.run()
        return timer


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')


def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)


_CONFIG_NONE = '__none__'


def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def load_config(path: Union[Path, str]) -> Any:
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))