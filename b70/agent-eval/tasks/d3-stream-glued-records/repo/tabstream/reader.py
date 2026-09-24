"""Records from a binary stream: a file, or a socket's makefile("rb")."""
import codecs

from .decoder import Decoder


def read_records(stream, chunk_size=65536):
    """Yield each record in `stream`, reading `chunk_size` bytes at a time."""
    # Incremental, so a UTF-8 character split between two reads comes out whole.
    utf8 = codecs.getincrementaldecoder("utf-8")()
    decoder = Decoder()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        yield from decoder.feed(utf8.decode(chunk))
    yield from decoder.feed(utf8.decode(b"", final=True))
    yield from decoder.close()
