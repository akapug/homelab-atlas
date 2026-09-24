import unittest

from syncer.client import RetryError, call_with_retry
from syncer.config import RetryPolicy


def flaky(failures, result="done"):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= failures:
            raise ConnectionError("connection reset")
        return result
    return fn, calls


class ClientTest(unittest.TestCase):
    def test_first_try(self):
        fn, calls = flaky(0)
        slept = []
        self.assertEqual(call_with_retry(fn, RetryPolicy(), sleep=slept.append), "done")
        self.assertEqual((len(calls), slept), (1, []))

    def test_retries_then_succeeds(self):
        fn, calls = flaky(2)
        slept = []
        result = call_with_retry(fn, RetryPolicy(max_attempts=4, delay=1.5), sleep=slept.append)
        self.assertEqual(result, "done")
        self.assertEqual(slept, [1.5, 1.5])

    def test_gives_up(self):
        fn, calls = flaky(10)
        slept = []
        with self.assertRaises(RetryError) as ctx:
            call_with_retry(fn, RetryPolicy(max_attempts=3, delay=1), sleep=slept.append)
        self.assertEqual(ctx.exception.attempts, 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(slept, [1, 1])

    def test_other_errors_not_retried(self):
        def fn():
            raise KeyError("bad payload")
        slept = []
        with self.assertRaises(KeyError):
            call_with_retry(fn, RetryPolicy(), sleep=slept.append)
        self.assertEqual(slept, [])


if __name__ == "__main__":
    unittest.main()
