# bkp

Describes backup jobs and prints the commands the backup runner executes for
them. One job per JSON file.

    python3 -m bkp plan JOB.json     # the tar command and the prune rule
    python3 -m bkp show JOB.json     # a short summary
    python3 -m bkp fmt JOB.json...   # rewrite files in the canonical form

A job file (see `examples/photos.json`):

    {
      "name": "photos",
      "source_dir": "/srv/photos",
      "exclude": ["*.tmp", "cache/"],
      "dest": "/mnt/backup/photos",
      "keep": 14
    }

`name`, `source_dir` and `dest` are required; `exclude` defaults to `[]` and
`keep` (how many archives to keep) to 7. Unknown keys are an error. The
format's next version is described in `docs/config-v2.md`.

## Tests

    python3 -m unittest discover -s tests
