"""Calls a flaky remote operation, retrying on network errors."""
import time

RETRYABLE = (ConnectionError, TimeoutError)


class RetryError(RuntimeError):
    def __init__(self, attempts, last):
        super().__init__(f"gave up after {attempts} attempts: {last}")
        self.attempts = attempts
        self.last = last


def retry_delays(policy):
    """Seconds to wait before each retry, in order (max_attempts - 1 of them)."""
    return [policy.delay] * (policy.max_attempts - 1)


def call_with_retry(fn, policy, sleep=time.sleep, log=None):
    """Return fn(), retrying RETRYABLE errors per `policy`; raise RetryError when out of attempts.

    Other exceptions propagate immediately.
    """
    delays = retry_delays(policy)
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except RETRYABLE as err:
            if attempt == policy.max_attempts:
                raise RetryError(attempt, err) from err
            wait = delays[attempt - 1]
            if log:
                log(f"attempt {attempt} failed ({err}); retrying in {wait:g}s")
            sleep(wait)
