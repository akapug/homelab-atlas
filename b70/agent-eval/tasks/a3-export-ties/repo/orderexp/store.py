"""SQLite-backed orders table."""
import sqlite3

COLUMNS = "id, ref, updated_at, total_cents"


class Store:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS orders ("
            " id INTEGER PRIMARY KEY,"
            " ref TEXT NOT NULL,"
            " updated_at TEXT NOT NULL,"
            " total_cents INTEGER NOT NULL)"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS orders_updated_at ON orders (updated_at)")

    def add(self, ref, updated_at, total_cents):
        """Insert one order; returns its id."""
        cur = self.db.execute(
            "INSERT INTO orders (ref, updated_at, total_cents) VALUES (?, ?, ?)",
            (ref, updated_at, total_cents),
        )
        self.db.commit()
        return cur.lastrowid

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM orders").fetchone()[0]

    def changed_since(self, since, limit):
        """Up to `limit` orders changed after `since` (None: from the beginning), oldest first,
        as (id, ref, updated_at, total_cents) tuples."""
        if since is None:
            sql, args = f"SELECT {COLUMNS} FROM orders ORDER BY updated_at LIMIT ?", (limit,)
        else:
            sql = f"SELECT {COLUMNS} FROM orders WHERE updated_at > ? ORDER BY updated_at LIMIT ?"
            args = (since, limit)
        return self.db.execute(sql, args).fetchall()
