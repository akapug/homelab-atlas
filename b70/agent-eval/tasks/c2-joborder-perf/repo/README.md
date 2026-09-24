# jobq

Works out the order to run a batch of jobs in. The deploy tooling runs
`jobq order` on the nightly job graph and diffs the result against the
previous night's, so the order must be stable.

    python3 -m jobq order examples/nightly.jobs   # job names, one per line
    python3 -m jobq check examples/nightly.jobs   # validate only

## Job files

One job per line, `NAME PRIORITY [DEP,DEP,...]`; `#` starts a comment. See
`examples/nightly.jobs`.

## The order

A job can run once every job it depends on has run. Of the jobs that can
run, the one with the highest priority goes next; equal priorities go in the
order the jobs are declared in the file. If dependencies form a cycle, jobq
exits 1 and names every job that could not be scheduled, sorted by name.

## Tests

    ./run_tests.sh
