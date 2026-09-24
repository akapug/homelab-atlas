"""The order to run jobs in."""


class CycleError(Exception):
    def __init__(self, names):
        super().__init__("dependency cycle among: " + ", ".join(names))
        self.names = names


def run_order(jobs):
    """Names of `jobs` in the order to run them.

    A job is ready once every job it depends on has run. The ready job with
    the highest priority runs next; among equal priorities, the one declared
    first. If some jobs can never become ready (they sit on or behind a
    dependency cycle), raises CycleError naming all of them, sorted.
    """
    done = []
    pending = list(jobs)
    while pending:
        ready = [j for j in pending if all(d in done for d in j.deps)]
        if not ready:
            raise CycleError(sorted(j.name for j in pending))
        best = ready[0]
        for j in ready[1:]:
            if j.priority > best.priority:
                best = j
        done.append(best.name)
        pending.remove(best)
    return done
