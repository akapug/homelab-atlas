# archive-logs

Small helper that cron runs nightly to archive application logs.

    bin/archive-logs SRC_DIR DEST_DIR

Every `*.log` file directly inside `SRC_DIR` is gzip-compressed into
`DEST_DIR/<name>.gz` (for example `app.log` becomes `app.log.gz`). The
originals are left in place. Each archived file is also recorded in
`DEST_DIR/archive.log`, and the script finishes with a summary line
`archived N files`.

Exit status: 0 on success, 2 on a usage error. Cron alerts on any
nonzero exit.

## Tests

    tests/run_tests.sh
