"""Loading, checking and saving job files."""
import json
import os

FORMAT_VERSION = 1


class ConfigError(Exception):
    """A job file that cannot be used; the message says which file and why."""


# key: (type, required, default)
FIELDS = {
    "name": (str, True, None),
    "source_dir": (str, True, None),
    "exclude": (list, False, []),
    "dest": (str, True, None),
    "keep": (int, False, 7),
}


def validate(data, where="job"):
    """A copy of `data` with defaults filled in; raises ConfigError if it is not a usable job."""
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a JSON object")
    version = data.get("version", FORMAT_VERSION)
    if version != FORMAT_VERSION:
        raise ConfigError(f"{where}: unsupported version {version!r}")
    unknown = sorted(set(data) - set(FIELDS) - {"version"})
    if unknown:
        raise ConfigError(f"{where}: unknown key {unknown[0]!r}")
    job = {}
    for key, (kind, required, default) in FIELDS.items():
        if key not in data:
            if required:
                raise ConfigError(f"{where}: missing required key {key!r}")
            job[key] = list(default) if isinstance(default, list) else default
        elif not isinstance(data[key], kind) or isinstance(data[key], bool):
            raise ConfigError(f"{where}: {key} must be a {kind.__name__}")
        else:
            job[key] = data[key]
    if not all(isinstance(p, str) for p in job["exclude"]):
        raise ConfigError(f"{where}: exclude must be a list of strings")
    if job["keep"] < 1:
        raise ConfigError(f"{where}: keep must be 1 or more")
    return job


def load(path):
    """The job in the file at `path`, checked, with defaults filled in."""
    try:
        with open(path) as f:
            data = json.load(f)
    except OSError as e:
        raise ConfigError(f"{path}: {e.strerror}") from None
    except ValueError as e:
        raise ConfigError(f"{path}: not valid JSON ({e})") from None
    return validate(data, path)


def dumps(job):
    """The canonical text of a job file: 2-space indent, sorted keys, trailing newline."""
    return json.dumps(dict(validate(job), version=FORMAT_VERSION), indent=2, sort_keys=True) + "\n"


def save(job, path):
    """Write `job` to `path` in the canonical form, replacing the file atomically."""
    text = dumps(job)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
