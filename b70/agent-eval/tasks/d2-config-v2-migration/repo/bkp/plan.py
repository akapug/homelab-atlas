"""The commands a job runs, as text: the runner executes them and `bkp plan` prints them."""
from shlex import quote


def plan(job):
    """Lines describing `job`: its name, the tar command, then the prune rule."""
    stem = f"{job['dest']}/{job['name']}"
    excludes = "".join(f" --exclude={quote(p)}" for p in job["exclude"])
    return [
        f"job {job['name']}",
        f"  tar -czf {quote(stem + '-{date}.tar.gz')}{excludes} -C {quote(job['source_dir'])} .",
        f"  prune {quote(stem + '-*.tar.gz')} keep {job['keep']}",
    ]
