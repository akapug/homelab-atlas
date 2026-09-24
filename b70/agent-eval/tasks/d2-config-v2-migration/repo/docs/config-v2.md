# Job config format v2

Status: agreed, not implemented yet.

The same job as in the README, in v2:

    {
      "version": 2,
      "name": "photos",
      "source": {"path": "/srv/photos", "exclude": ["*.tmp", "cache/"]},
      "target": "/mnt/backup/photos",
      "keep": 14,
      "compression": "zstd"
    }

## Changes from v1

- `dest` is renamed `target`.
- `source_dir` and `exclude` move into a `source` object, as `path`
  (required) and `exclude` (optional, default `[]`).
- `compression` is new and required. It picks the archive format:

  | compression | tar flags    | archive suffix |
  |-------------|--------------|----------------|
  | `gzip`      | `-czf`       | `.tar.gz`      |
  | `zstd`      | `--zstd -cf` | `.tar.zst`     |
  | `none`      | `-cf`        | `.tar`         |

- `version` is required and must be `2`. `name` and `keep` do not change.

## Compatibility

- v1 files (no `version` key, or `"version": 1`) keep loading. They are
  converted to v2 when loaded, with `compression` set to `gzip`, which is what
  every v1 job does today, so a v1 job's plan does not change.
- Loading returns the v2 shape whatever version the file is in, so the code
  that uses a job only ever reads v2 keys.
- Saving, and so `bkp fmt`, always writes v2 in the canonical form used today
  (2-space indent, sorted keys, trailing newline). `bkp fmt old.json` is how
  people migrate their files.
- A v2 file must be complete: a v2 file without `compression`, or with a v1
  key such as `dest`, is an error, not something to fill in or convert.
- Any other `version` is an error.
