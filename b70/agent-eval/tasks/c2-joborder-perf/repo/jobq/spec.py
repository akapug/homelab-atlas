"""Job files.

One job per line: `NAME PRIORITY [DEP,DEP,...]`. A higher PRIORITY runs
sooner. DEPs are the names of jobs that must finish first; they may be
declared anywhere in the file. `#` starts a comment.
"""
from dataclasses import dataclass


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class Job:
    name: str
    priority: int
    deps: tuple = ()


def parse(text):
    """The jobs in `text`, in the order they are declared."""
    jobs, names = [], []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) not in (2, 3):
            raise SpecError(f"line {n}: expected NAME PRIORITY [DEPS], got {raw.strip()!r}")
        name, prio = fields[0], fields[1]
        try:
            priority = int(prio)
        except ValueError:
            raise SpecError(f"line {n}: bad priority {prio!r}") from None
        if name in names:
            raise SpecError(f"line {n}: duplicate job {name}")
        names.append(name)
        deps = tuple(fields[2].split(",")) if len(fields) == 3 else ()
        jobs.append(Job(name, priority, deps))
    for job in jobs:
        for dep in job.deps:
            if dep not in names:
                raise SpecError(f"job {job.name}: unknown dependency {dep!r}")
    return jobs
