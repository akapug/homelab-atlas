# Read GGUF headers only (no tensor data); print JSON {files, kv, tensors}: rows name, dims, type, bytes (from offsets).
# Kept for the commands quoted in NOTES.md; the reader lives in probe/gguf.py. Reads exactly the files given (no shard
# expansion), as this script always did; `python3 probe/gguf.py FIRST-SHARD` expands a split model itself.
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'probe'))
import gguf
json.dump(gguf.read_files(sys.argv[1:], shards=False), sys.stdout)
