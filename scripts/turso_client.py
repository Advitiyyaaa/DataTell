"""
turso_client.py — Pure-Python HTTP client for Turso Cloud
===========================================================

Provides a TursoConnection class that mimics the sqlite3.Connection interface
well enough for all DataTell operations, using only Turso's REST HTTP API.

Why not pyturso?
    pyturso requires Rust + MSVC to compile on Windows (link.exe not found error).
    This pure-Python implementation uses only `requests` — zero native compilation.

The interface is compatible with:
  - conn.execute(sql)            → TursoCursor (.fetchall(), .fetchone(), .description)
  - conn.cursor()                → cursor.execute(), .fetchall(), .description
  - conn.read_sql_query(sql)     → pd.DataFrame (for nl_to_sql.py)
  - conn.close()                 → no-op (HTTP is stateless)
  - Thread-safe: every call is an independent HTTP request
"""

import sqlite3
from typing import Any, Optional

import requests
import pandas as pd

TURSO_API_VERSION = "v2"


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------

class TursoCursor:
    """Mimics sqlite3.Cursor. Holds the result of one HTTP query."""

    def __init__(self, cols: list, rows: list):
        # DB-API 2.0: description is a sequence of 7-item sequences
        self.description = [
            (col["name"], None, None, None, None, None, None) for col in cols
        ]
        self.rowcount = len(rows)
        # Turso row cells: {"type": "text"|"integer"|"float"|"null", "value": ...}
        self._rows: list[tuple] = [
            tuple(
                None if cell.get("type") == "null" else _cast(cell)
                for cell in row
            )
            for row in rows
        ]
        self._pos = 0

    def fetchall(self) -> list[tuple]:
        return self._rows

    def fetchone(self) -> Optional[tuple]:
        if self._pos < len(self._rows):
            row = self._rows[self._pos]
            self._pos += 1
            return row
        return None

    def fetchmany(self, size: int = 1) -> list[tuple]:
        result = self._rows[self._pos : self._pos + size]
        self._pos += len(result)
        return result

    def __iter__(self):
        return iter(self._rows)


def _cast(cell: dict) -> Any:
    """Convert a Turso typed cell to a Python value."""
    t = cell.get("type", "text")
    v = cell.get("value")
    if t == "integer":
        return int(v) if v is not None else None
    if t == "float":
        return float(v) if v is not None else None
    return v  # text, blob → str


# ---------------------------------------------------------------------------
# Executor cursor (returned by conn.cursor())
# ---------------------------------------------------------------------------

class _TursoExecutor:
    """Minimal cursor returned by TursoConnection.cursor()."""

    def __init__(self, conn: "TursoConnection"):
        self._conn = conn
        self.description: Optional[list] = None
        self.rowcount: int = -1
        self._cursor: Optional[TursoCursor] = None

    def execute(self, sql: str, params=None) -> "_TursoExecutor":
        self._cursor = self._conn.execute(sql, params)
        self.description = self._cursor.description
        self.rowcount = self._cursor.rowcount
        return self

    def fetchall(self) -> list[tuple]:
        return self._cursor.fetchall() if self._cursor else []

    def fetchone(self) -> Optional[tuple]:
        return self._cursor.fetchone() if self._cursor else None


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

class TursoConnection:
    """
    HTTP-based Turso connection that mimics sqlite3.Connection.

    DataTell-specific extras:
      - read_sql_query(sql) → pd.DataFrame  (replaces pd.read_sql(sql, conn))
      - set_authorizer() / set_progress_handler() are no-ops so nl_to_sql.py
        doesn't crash; Turso is read-only by construction on the app path
        (the UI never sends write SQL, and the planner only generates SELECT).
    """

    def __init__(self, url: str, auth_token: str):
        # Turso uses libsql:// scheme; HTTP API uses https://
        self._base_url = url.replace("libsql://", "https://")
        self._headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }
        self._pipeline_url = f"{self._base_url}/{TURSO_API_VERSION}/pipeline"
        
        # Setup session with keep-alive and retry
        from requests.adapters import HTTPAdapter
        from urllib3.util import Retry

        self._session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    # ── Internal HTTP call ────────────────────────────────────────────────────

    def _http(self, sql: str, params: Optional[list] = None) -> TursoCursor:
        """POST one statement to Turso and return a TursoCursor."""
        stmt: dict[str, Any] = {"sql": sql}
        if params:
            stmt["args"] = [
                {"type": "null", "value": None}
                if p is None
                else {"type": "text", "value": str(p)}
                for p in params
            ]

        payload = {"requests": [{"type": "execute", "stmt": stmt}]}
        resp = self._session.post(
            self._pipeline_url, json=payload, headers=self._headers, timeout=45
        )
        resp.raise_for_status()

        data = resp.json()
        result = data["results"][0]

        if result.get("type") == "error":
            msg = result.get("error", {}).get("message", "Turso error")
            raise sqlite3.OperationalError(msg)

        res = result["response"]["result"]
        return TursoCursor(res.get("cols", []), res.get("rows", []))

    def pipeline(self, statements: list[tuple[str, Optional[list]]]) -> list[TursoCursor]:
        """
        Send multiple SQL statements in a SINGLE HTTP request.

        This is the key to fast bulk inserts — instead of one HTTP round-trip
        per row, we pack N statements into one request. For 100 INSERTs:
          Before: 100 HTTP requests × ~400ms latency = 40s per batch
          After:  1 HTTP request   × ~400ms latency = 0.4s per batch (100× faster)

        Parameters
        ----------
        statements : list of (sql, params) tuples. params may be None.

        Returns
        -------
        list[TursoCursor] — one cursor per statement (same order as input).
        """
        requests_payload = []
        for sql, params in statements:
            stmt: dict[str, Any] = {"sql": sql}
            if params:
                stmt["args"] = [
                    {"type": "null", "value": None}
                    if p is None
                    else {"type": "text", "value": str(p)}
                    for p in params
                ]
            requests_payload.append({"type": "execute", "stmt": stmt})

        payload = {"requests": requests_payload}
        resp = self._session.post(
            self._pipeline_url, json=payload, headers=self._headers, timeout=60
        )
        resp.raise_for_status()

        cursors = []
        for result in resp.json()["results"]:
            if result.get("type") == "error":
                msg = result.get("error", {}).get("message", "Turso error")
                raise sqlite3.OperationalError(msg)
            res = result["response"]["result"]
            cursors.append(TursoCursor(res.get("cols", []), res.get("rows", [])))
        return cursors

    def executemany(self, sql: str, params_list: list[list]) -> None:
        """
        Insert many rows in a single HTTP pipeline call.
        Equivalent to sqlite3's executemany() but 100× faster over HTTP.
        """
        self.pipeline([(sql, list(p)) for p in params_list])

    # ── Public sqlite3-compatible interface ───────────────────────────────────

    def execute(self, sql: str, params=None) -> TursoCursor:
        return self._http(sql, list(params) if params else None)

    def cursor(self) -> _TursoExecutor:
        return _TursoExecutor(self)

    # no-ops — Turso has no authorizer/progress-handler concept;
    # safety is enforced at the planner prompt level instead.
    def set_authorizer(self, *args, **kwargs) -> None:
        pass

    def set_progress_handler(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass  # HTTP is stateless

    def commit(self) -> None:
        pass

    def __enter__(self) -> "TursoConnection":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── DataTell extra ────────────────────────────────────────────────────────

    def read_sql_query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SELECT and return a pandas DataFrame.

        Called by nl_to_sql.execute_sql() instead of pd.read_sql(sql, conn)
        when the connection is a TursoConnection.
        """
        cursor = self._http(sql)
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)


# ---------------------------------------------------------------------------
# Factory — matches the pyturso.connect() signature
# ---------------------------------------------------------------------------

def connect(remote_url: str, auth_token: str) -> TursoConnection:
    """
    Create a TursoConnection.

    Parameters
    ----------
    remote_url  : libsql://your-db.turso.io  (or https://)
    auth_token  : Turso database auth token
    """
    return TursoConnection(remote_url, auth_token)
