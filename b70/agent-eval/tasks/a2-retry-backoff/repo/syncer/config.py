"""Settings: built-in defaults < config file < environment variables."""
import configparser
import os
from dataclasses import dataclass, field


class ConfigError(ValueError):
    pass


@dataclass
class RetryPolicy:
    max_attempts: int = 4
    delay: float = 2.0


@dataclass
class Config:
    endpoint: str = "http://localhost:8080/api"
    timeout: float = 10.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)


# (file section, key, environment variable, type)
FIELDS = [
    ("sync", "endpoint", "SYNC_ENDPOINT", str),
    ("sync", "timeout", "SYNC_TIMEOUT", float),
    ("retry", "max_attempts", "SYNC_RETRY_MAX_ATTEMPTS", int),
    ("retry", "delay", "SYNC_RETRY_DELAY", float),
]


def load_config(path=None, env=None):
    """The effective Config. `env` defaults to os.environ."""
    env = os.environ if env is None else env
    parser = configparser.ConfigParser()
    if path and not parser.read(path):
        raise ConfigError(f"cannot read config file {path}")
    cfg = Config()
    for section, key, var, kind in FIELDS:
        raw = env.get(var, parser.get(section, key, fallback=None))
        if raw is None:
            continue
        try:
            value = kind(raw)
        except ValueError:
            raise ConfigError(f"bad value for {section}.{key}: {raw!r}") from None
        setattr(cfg.retry if section == "retry" else cfg, key, value)
    validate(cfg)
    return cfg


def validate(cfg):
    if cfg.timeout <= 0:
        raise ConfigError("sync.timeout must be positive")
    if cfg.retry.max_attempts < 1:
        raise ConfigError("retry.max_attempts must be at least 1")
    if cfg.retry.delay < 0:
        raise ConfigError("retry.delay must not be negative")
