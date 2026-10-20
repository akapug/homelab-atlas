#!/usr/bin/env python3
"""GGUF header reader: metadata + tensor table, never the tensor data. Standard library only.

    python3 gguf.py [--no-shards] FILE.gguf [...]      -> JSON {files, kv, tensors} on stdout

FILE may be the first shard of a split model ("...-00001-of-00010.gguf"): the other shards' headers are read too
(header bytes only), unless --no-shards. The file can also be piped to a remote host and run there
(`ssh host python3 - FILE < gguf.py`), which reads only the header bytes on the host.

Output (the format src/hdr-*.json use):
  files:   [[basename, file size, tensor count], ...]
  kv:      {key: value}; arrays longer than 64 entries (or string arrays longer than 16) become "[arr <type> <n>]"
           ("[arr str <n>]" string arrays are left out), first shard's value wins
  tensors: [[name, dims, ggml type id, bytes, basename], ...]; bytes = distance to the next tensor's offset (the last
           tensor: to the end of the file), so padding is included and every type, known or not, gets a size
Library use: read_header(fetch, size) with fetch(offset, n) -> bytes (local file, ssh, HTTP range requests).
"""
import json
import os
import re
import struct
import sys

# scalar GGUF value types: id -> (struct format, size)
SCALAR = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2), 4: ('I', 4), 5: ('i', 4), 6: ('f', 4), 7: ('?', 1), 10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}
STR, ARR = 8, 9

# ggml type ids -> name. Mainline ids 0-42 (ggml.h); ids >= 97 are ik_llama.cpp's own (ggml/include/ggml.h there).
# Ids 34/35 differ between forks (mainline TQ1_0/TQ2_0); a name here is a label only: sizes never depend on it.
GGML_TYPES = {0: 'f32', 1: 'f16', 2: 'q4_0', 3: 'q4_1', 6: 'q5_0', 7: 'q5_1', 8: 'q8_0', 9: 'q8_1', 10: 'q2_K', 11: 'q3_K', 12: 'q4_K',
              13: 'q5_K', 14: 'q6_K', 15: 'q8_K', 16: 'iq2_xxs', 17: 'iq2_xs', 18: 'iq3_xxs', 19: 'iq1_s', 20: 'iq4_nl', 21: 'iq3_s', 22: 'iq2_s',
              23: 'iq4_xs', 24: 'i8', 25: 'i16', 26: 'i32', 27: 'i64', 28: 'f64', 29: 'iq1_m', 30: 'bf16', 34: 'tq1_0', 35: 'tq2_0', 36: 'i2_s',
              39: 'mxfp4', 41: 'q1_0_g128', 42: 'q2_0',
              133: 'q6_0', 134: 'iq1_bn', 135: 'iq2_bn', 137: 'iq2_k', 138: 'iq3_k', 139: 'iq4_k', 140: 'iq5_k', 141: 'iq6_k', 142: 'pq2_0',
              143: 'ptq1_0', 144: 'iq4_ks', 145: 'iq2_ks', 146: 'iq4_kss', 152: 'iq5_ks', 153: 'iq2_kt', 154: 'iq3_kt', 155: 'iq4_kt', 156: 'iq3_ks',
              157: 'iq2_kl', 158: 'iq1_kt', 202: 'q4_0_r8', 206: 'q5_0_r4', 208: 'q8_0_r8', 210: 'q2_K_r4', 211: 'q3_K_r4', 212: 'q4_K_r4',
              213: 'q5_K_r4', 214: 'q6_K_r4', 220: 'iq4_nl_r4', 223: 'iq4_xs_r8', 230: 'bf16_r16', 344: 'iq4_ks_r4', 353: 'mxfp4_r8'}
SHARD = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$')


def type_name(t):
    return GGML_TYPES.get(t, 'type%d' % t)


class Reader:
    """Sequential reader over fetch(offset, n). Pulls `chunk` bytes at a time, doubling up to 8 MiB, so a small header
    costs one small read and a large one a few range requests; at most one chunk past the header end is read."""

    def __init__(self, fetch, size=None, chunk=64 << 10):
        self.fetch, self.size, self.chunk, self.buf, self.base, self.pos = fetch, size, chunk, b'', 0, 0

    def read(self, n):
        need = self.pos + n - (self.base + len(self.buf))
        if need > 0:
            more = self.fetch(self.base + len(self.buf), max(self.chunk, need))
            if len(more) < need:
                raise EOFError('header runs past the end of the data (%d bytes short)' % (need - len(more)))
            self.buf += more
            self.chunk = min(self.chunk * 2, 8 << 20)  # headers range from KiB to tens of MiB (tokenizers)
        off = self.pos - self.base
        self.pos += n
        return self.buf[off:off + n]

    def u32(self):
        return struct.unpack('<I', self.read(4))[0]

    def u64(self):
        return struct.unpack('<Q', self.read(8))[0]

    def s(self):
        return self.read(self.u64()).decode('utf8', 'replace')


def value(r, t):
    if t == STR:
        return r.s()
    if t == ARR:
        et, cnt = r.u32(), r.u64()
        if et == STR:
            a = [r.s() for _ in range(cnt)]
            return a if cnt <= 16 else '[arr str %d]' % cnt
        if et == ARR:
            raise ValueError('nested GGUF arrays are not supported')
        fmt, sz = SCALAR[et]
        b = r.read(sz * cnt)
        if cnt <= 64:
            return list(struct.unpack('<%d%s' % (cnt, fmt), b))
        return '[arr %d %d]' % (et, cnt)
    fmt, sz = SCALAR[t]
    return struct.unpack('<' + fmt, r.read(sz))[0]


def read_header(fetch, size, chunk=64 << 10):
    """-> {'version', 'kv', 'tensors': [[name, dims, type, offset]], 'data_start', 'size'}"""
    r = Reader(fetch, size, chunk)
    if r.read(4) != b'GGUF':
        raise ValueError('not a GGUF file')
    ver = r.u32()
    nt, nkv = r.u64(), r.u64()
    kv, align = {}, 32
    for _ in range(nkv):
        k = r.s()
        v = value(r, r.u32())
        if k == 'general.alignment':
            align = v
        kv[k] = v
    ts = []
    for _ in range(nt):
        name = r.s()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        t = r.u32()
        off = r.u64()
        ts.append([name, dims, t, off])
    ds = (r.pos + align - 1) // align * align
    return {'version': ver, 'kv': kv, 'tensors': ts, 'data_start': ds, 'size': size}


def sizes(h):
    """tensor rows with bytes from the offset table: [[name, dims, type, bytes]]"""
    ts = sorted(h['tensors'], key=lambda x: x[3])
    out = []
    for i, (n, d, t, o) in enumerate(ts):
        nxt = ts[i + 1][3] if i + 1 < len(ts) else h['size'] - h['data_start']
        out.append([n, d, t, nxt - o])
    return out


def shard_paths(path):
    m = SHARD.match(path)
    if not m:
        return [path]
    total = int(m.group(3))
    return ['%s-%05d-of-%05d.gguf' % (m.group(1), i, total) for i in range(1, total + 1)]


def merge(headers):
    """[(basename, header)] -> the {files, kv, tensors} summary"""
    out = {'files': [], 'kv': {}, 'tensors': []}
    for base, h in headers:
        for k, v in h['kv'].items():
            if not str(v).startswith('[arr str') and k not in out['kv']:
                out['kv'][k] = v
        out['tensors'] += [row + [base] for row in sizes(h)]
        out['files'].append([base, h['size'], len(h['tensors'])])
    return out


def local_fetch(f):
    def fetch(off, n):
        f.seek(off)
        return f.read(n)
    return fetch


def read_files(paths, shards=True):
    todo = []
    for p in paths:
        todo += [x for x in (shard_paths(p) if shards else [p]) if x not in todo]
    hs = []
    for p in todo:
        with open(p, 'rb') as f:
            hs.append((os.path.basename(p), read_header(local_fetch(f), os.path.getsize(p))))
    return merge(hs)


if __name__ == '__main__':
    args = sys.argv[1:]
    shards = '--no-shards' not in args
    files = [a for a in args if not a.startswith('--')]
    if not files:
        sys.exit(__doc__)
    json.dump(read_files(files, shards), sys.stdout)
