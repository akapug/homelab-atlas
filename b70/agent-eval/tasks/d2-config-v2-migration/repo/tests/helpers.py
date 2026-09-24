import json
import os
import tempfile


def job_file(test, data, name="job.json"):
    """Write `data` as JSON to a temp file removed after `test`; returns its path."""
    d = tempfile.TemporaryDirectory()
    test.addCleanup(d.cleanup)
    path = os.path.join(d.name, name)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "photos.json")
