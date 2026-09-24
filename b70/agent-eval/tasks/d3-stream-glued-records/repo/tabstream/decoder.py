"""Incremental decoding: text arrives in pieces, records come out."""
from .escapes import decode_record


class Decoder:
    """Feed text in pieces of any size; get back the records each piece completes.

    A record ends at a bare LF. A backslash escapes the character after it, so the scan steps over
    escape pairs: an escaped character never ends a record. Records can be very large, so the
    decoder remembers how far it has scanned the unfinished record and never scans text twice.
    """

    def __init__(self):
        self._pending = ""  # the unfinished record so far
        self._scanned = 0   # how much of _pending has been scanned for the record's end

    def feed(self, text):
        """The records completed by `text`, in order."""
        data = self._pending + text
        records = []
        start = 0
        i = self._scanned
        while i < len(data):
            c = data[i]
            if c == "\\":
                i += 2
            elif c == "\n":
                records.append(decode_record(data[start:i]))
                i = start = i + 1
            else:
                i += 1
        self._pending = data[start:]
        self._scanned = len(self._pending)
        return records

    def close(self):
        """The last record, if the stream did not end with LF."""
        rest = self._pending
        self._pending, self._scanned = "", 0
        return [decode_record(rest)] if rest else []


def decode_text(text):
    """Every record in a complete text."""
    decoder = Decoder()
    return decoder.feed(text) + decoder.close()
