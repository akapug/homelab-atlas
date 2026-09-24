# tabstream

Decodes the record stream our exporter sends to the collector. The format is
the text format of PostgreSQL's COPY: each record is its fields joined by TAB
and ended by LF. Inside a field a backslash starts a two-character escape:
`\\` is a backslash, `\t` a tab, `\n` a line feed and `\r` a carriage return.
A field that is exactly `\N` is NULL (None). The bytes are UTF-8.

    from tabstream.reader import read_records
    for record in read_records(sock.makefile("rb")):   # or any binary file
        ...

`Decoder` (in `tabstream/decoder.py`) is the incremental part: feed it text in
pieces of any size and it returns the records each piece completes. Records
can be very large (whole documents travel as fields), so it never rescans
text it has already looked at.

    python3 -m tabstream FILE    # print each record of FILE as a JSON array

## Tests

    python3 -m unittest discover -s tests
