"""
agent_router.py -- Phase 4
==========================
LangGraph-based query router that intelligently routes questions to:
  - SQL tool   : nl_to_sql() for quantitative / transactional queries
  - RAG tool   : rag_query() for policy / knowledge base queries
  - Both tools : compound queries needing data + policy (parallel fan-out)
  - Synthesizer: chitchat / merge compound results into one coherent answer

Graph topology:
                     +- sql_tool ------+
  planner --(route)--+- rag_tool ------+--> synthesizer --> END
                     +- both_tools ----+
                     +- synthesizer (chitchat shortcut)

Public API:
    load_resources()                          -> (conn, schema, collection, bm25, chunks)
    build_agent_graph(conn, schema, ...)      -> CompiledGraph
    run_agent(question, graph)                -> dict

Usage (from project root):
    python scripts/agent_router.py "What are the top products by revenue?"
"""

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
_api_key = os.getenv("GEMINI_API_KEY")

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Mutable state passed between LangGraph nodes."""
    question:     str
    route:        Optional[str]   # "SQL" | "RAG" | "BOTH" | "CHITCHAT"
    route_reason: Optional[str]   # planner one-sentence reasoning
    sql_result:   Optional[dict]  # full dict from nl_to_sql()
    rag_result:   Optional[dict]  # full dict from rag_query()
    final_answer: Optional[str]
    error:        Optional[str]
    metadata:     dict            # timing, model call info, etc.


# ---------------------------------------------------------------------------
# Planner Node -- classify query into a route
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are the query router for DataTell, an analytics assistant for the Olist \
Brazilian e-commerce platform (2016-2018).

You have access to two information sources:
1. SQL DATABASE -- Transactional data: orders, customers, sellers, products, \
   order_items, payments, reviews. Use for questions about counts, totals, \
   rankings, trends, averages, delivery times, review scores, and any \
   other quantitative or data-driven question.
2. RAG DOCUMENTS -- Olist policy documents: return policy, shipping policy, \
   seller guidelines, customer FAQ. Use for questions about rules, procedures, \
   eligibility criteria, legal obligations, or anything policy-related.

Classify the user question into exactly one route:
- SQL     : answerable purely from the transactional database
- RAG     : answerable purely from the policy documents
- BOTH    : requires BOTH database data AND policy document context
- CHITCHAT: greeting, meta-question about the system, or completely unanswerable

EXAMPLES:
  "What are the top 5 product categories by revenue?"                        -> SQL
  "How many orders were delivered late in 2017?"                             -> SQL
  "Which sellers have the highest average review score?"                     -> SQL
  "What is the return window for defective products?"                        -> RAG
  "What are the seller performance guidelines?"                              -> RAG
  "How does the refund process work?"                                        -> RAG
  "What is the average order value for electronics and the return policy?"   -> BOTH
  "How many late deliveries in 2017 and what are the seller delay guidelines?" -> BOTH
  "Which top sellers have the best performance per Olist evaluation criteria?" -> BOTH
  "Hello, what can you do?"                                                  -> CHITCHAT
  "What is the capital of France?"                                           -> CHITCHAT

CRITICAL: Respond with ONLY valid JSON -- no preamble, no markdown fences:
{"route": "<SQL|RAG|BOTH|CHITCHAT>", "reasoning": "<one sentence>"}
"""


def _is_daily_quota_exhausted(exc_str: str) -> bool:
    """Return True when the error is a daily (not per-minute) quota exhaustion."""
    return "GenerateRequestsPerDayPerProjectPerModel" in exc_str or "PerDay" in exc_str


def _classify_question(question: str, model: str = "gemini-3.1-flash-lite") -> dict:
    """
    Call Gemini to classify the question.
    Returns {"route": str, "reasoning": str}.

    Rate-limit handling:
    - Per-minute (429 without daily flag): retry with exponential backoff
    - Daily quota exhausted: raise immediately (backoff won't help)
    """
    client = genai.Client(api_key=_api_key)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=f"Question: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=_PLANNER_SYSTEM,
                    temperature=0.0,
                ),
            )
            text = response.text.strip()
            # Strip markdown code fences if the model adds them
            if text.startswith("```"):
                inner = text.split("```")
                text = inner[1] if len(inner) > 1 else inner[0]
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            return json.loads(text.strip())
        except json.JSONDecodeError:
            if attempt == 2:
                return {"route": "CHITCHAT", "reasoning": "Failed to parse planner response"}
        except Exception as exc:
            err = str(exc)
            if "429" in err:
                if _is_daily_quota_exhausted(err):
                    raise RuntimeError(
                        f"Daily API quota exhausted for model '{model}'. "
                        "Wait until tomorrow or use a different model/API key."
                    ) from exc
                if attempt < 2:
                    wait = 35 * (attempt + 1)
                    print(f"  [Planner] Rate limited -- waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            else:
                raise
    return {"route": "CHITCHAT", "reasoning": "Planner exhausted retries"}


def planner_node(state: AgentState) -> dict:
    """
    Planner node: classify the question and write the routing decision.
    State updates: route, route_reason, metadata.planner_ms
    """
    t0 = time.time()
    result = _classify_question(state["question"])
    route  = result.get("route", "CHITCHAT").upper().strip()
    if route not in {"SQL", "RAG", "BOTH", "CHITCHAT"}:
        route = "CHITCHAT"
    return {
        "route":        route,
        "route_reason": result.get("reasoning", ""),
        "metadata": {
            **state.get("metadata", {}),
            "planner_ms": round((time.time() - t0) * 1000),
        },
    }


def route_after_planner(state: AgentState) -> str:
    """Conditional edge function: select next node by route string."""
    return state.get("route", "CHITCHAT")


# ---------------------------------------------------------------------------
# SQL Tool Node -- wraps nl_to_sql()
# ---------------------------------------------------------------------------

def make_sql_tool_node(conn, schema):
    """
    Factory capturing (conn, schema) in a closure.
    Returns a LangGraph-compatible node function.
    """
    def sql_tool_node(state: AgentState) -> dict:
        scripts_dir = str(Path(__file__).parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from nl_to_sql import nl_to_sql

        t0 = time.time()
        try:
            result = nl_to_sql(
                state["question"],
                schema,
                conn,
                model="gemini-3.1-flash-lite",
                max_retries=3,
                verbose=False,
            )
        except Exception as exc:
            result = {
                "question": state["question"],
                "sql":       None,
                "results":   None,
                "success":   False,
                "error":     str(exc),
                "attempts":  0,
                "row_count": 0,
                "truncated": False,
            }
        result["tool_ms"] = round((time.time() - t0) * 1000)
        return {"sql_result": result}

    return sql_tool_node


# ---------------------------------------------------------------------------
# RAG Tool Node -- wraps rag_query()
# ---------------------------------------------------------------------------

def make_rag_tool_node(collection, bm25, chunks):
    """
    Factory capturing (collection, bm25, chunks) in a closure.
    Returns a LangGraph-compatible node function.
    """
    def rag_tool_node(state: AgentState) -> dict:
        scripts_dir = str(Path(__file__).parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from rag_pipeline import rag_query

        t0 = time.time()
        try:
            result = rag_query(
                state["question"],
                collection,
                bm25,
                chunks,
                model="gemini-3.1-flash-lite",
                k=5,
                verbose=False,
            )
        except Exception as exc:
            result = {
                "question":   state["question"],
                "answer":     None,
                "chunks":     [],
                "sources":    [],
                "answerable": False,
                "error":      str(exc),
            }
        result["tool_ms"] = round((time.time() - t0) * 1000)
        return {"rag_result": result}

    return rag_tool_node


# ---------------------------------------------------------------------------
# Both Tools Node -- parallel SQL + RAG for compound queries
# ---------------------------------------------------------------------------

def make_both_tools_node(conn, schema, collection, bm25, chunks, db_path: Optional[Path] = None):
    """
    Factory: runs SQL and RAG concurrently via ThreadPoolExecutor.

    Why threads (not async): nl_to_sql and rag_query make blocking
    API/DB calls. Threads let them run in parallel without rewriting them
    as coroutines. Wall-clock ~ max(sql_time, rag_time) not sql + rag.

    Thread-safety note: SQLite connections must NOT be shared across threads.
    We create a fresh, short-lived connection inside the SQL worker thread and
    close it when done. The RAG path (ChromaDB + BM25) is already thread-safe.
    """
    import sqlite3 as _sqlite3

    # Resolve DB path: use explicit arg, else infer from project layout
    _db_path = db_path or (Path(__file__).parent.parent / "db" / "analytics.db")
    _rag_fn  = make_rag_tool_node(collection, bm25, chunks)

    def _run_sql_in_thread(state: AgentState) -> dict:
        """Open a thread-local connection, run SQL, close it."""
        thread_conn = _sqlite3.connect(str(_db_path))
        try:
            sql_fn = make_sql_tool_node(thread_conn, schema)
            return sql_fn(state)
        finally:
            thread_conn.close()

    def both_tools_node(state: AgentState) -> dict:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            sql_future = executor.submit(_run_sql_in_thread, state)
            rag_future = executor.submit(_rag_fn, state)
            sql_update = sql_future.result()  # raises on worker exception
            rag_update = rag_future.result()
        return {**sql_update, **rag_update}

    return both_tools_node



# ---------------------------------------------------------------------------
# Synthesizer Node -- merge evidence into a final answer
# ---------------------------------------------------------------------------

_SYNTHESIZER_SYSTEM = """\
You are the answer synthesizer for DataTell, an analytics assistant for \
Olist e-commerce. Compose a clear, accurate, well-structured answer from \
the provided evidence.

Formatting rules:
1. Integrate all evidence into ONE coherent answer.
2. SQL data: present in a neat table or bulleted list. Include column names naturally.
   If SQL failed, acknowledge it gracefully (no raw error messages).
3. RAG policy text: summarize or quote accurately. Cite the source document name.
4. For compound answers (SQL + RAG): use markdown headers to separate the data \
   section from the policy section, then add a brief "Summary" paragraph tying both.
5. Never fabricate information not present in the evidence.
6. Keep responses concise but complete.
"""

_CHITCHAT_SYSTEM = """\
You are DataTell, a friendly analytics assistant for the Olist Brazilian \
e-commerce platform. You help users with:
- Transactional data queries: orders, revenue, customers, sellers, delivery \
  times, review scores (live SQLite database, 2016-2018 Olist data)
- Platform policy questions: returns, shipping, seller guidelines, customer \
  FAQ (document knowledge base)
- Compound queries combining data + policy in a single answer

Reply naturally and helpfully. If asked what you can do, describe these \
capabilities briefly with a friendly example.
"""


def _df_to_table_str(df) -> str:
    """Format a DataFrame as a readable string (markdown table if tabulate available)."""
    try:
        return df.head(20).to_markdown(index=False)
    except Exception:
        return df.head(20).to_string(index=False)


def _format_sql_for_synthesis(sql_result: dict) -> str:
    """Convert a sql_result dict to a readable evidence string."""
    if not sql_result:
        return "[SQL: no result returned]"
    if not sql_result.get("success"):
        err = sql_result.get("error", "unknown error")
        return f"[SQL query failed: {err}]"

    df = sql_result.get("results")
    if df is None or len(df) == 0:
        return f"[SQL ran successfully but returned no rows]\nSQL: {sql_result.get('sql', '')}"

    row_count  = sql_result.get("row_count", len(df))
    truncated  = sql_result.get("truncated", False)
    trunc_note = f" (showing first {len(df)} of {row_count})" if truncated else ""
    table_str  = _df_to_table_str(df)

    return (
        f"SQL Query Results ({row_count} rows{trunc_note}):\n\n"
        f"{table_str}\n\n"
        f"SQL used:\n```sql\n{sql_result.get('sql', '')}\n```"
    )


def _format_rag_for_synthesis(rag_result: dict) -> str:
    """Convert a rag_result dict to a readable evidence string."""
    if not rag_result:
        return "[RAG: no result returned]"
    if not rag_result.get("answerable", False):
        return "[RAG: no relevant policy information found]"

    answer  = rag_result.get("answer", "")
    sources = rag_result.get("sources", [])
    src_str = ", ".join(sources) if sources else "policy documents"
    return f"Policy Knowledge Base (sources: {src_str}):\n\n{answer}"


def make_synthesizer_node(model: str = "gemini-3.1-flash-lite"):
    """
    Factory: returns a synthesizer node.

    Handles four scenarios:
    - CHITCHAT : conversational Gemini reply (no evidence)
    - RAG only : pass through rag_query answer directly (already generated)
    - SQL only : Gemini narrates the SQL result into clear prose
    - BOTH     : Gemini weaves SQL table + RAG text into one answer
    """

    def synthesizer_node(state: AgentState) -> dict:
        route    = state.get("route", "CHITCHAT")
        question = state["question"]

        # CHITCHAT: conversational reply
        if route == "CHITCHAT":
            client = genai.Client(api_key=_api_key)
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=question,
                    config=types.GenerateContentConfig(
                        system_instruction=_CHITCHAT_SYSTEM,
                        temperature=0.5,
                    ),
                )
                return {"final_answer": response.text}
            except Exception as exc:
                return {
                    "final_answer": "Hi! I am DataTell, your Olist analytics assistant.",
                    "error": str(exc),
                }

        sql_result = state.get("sql_result")
        rag_result = state.get("rag_result")

        # RAG only: answer already synthesized by rag_query() -- pass through
        if route == "RAG" and rag_result is not None:
            answer = rag_result.get("answer") or "No relevant policy information found."
            return {"final_answer": answer}

        # SQL only or BOTH: build evidence block and call Gemini
        evidence_parts = []
        if sql_result is not None:
            evidence_parts.append(_format_sql_for_synthesis(sql_result))
        if rag_result is not None:
            evidence_parts.append(_format_rag_for_synthesis(rag_result))

        if not evidence_parts:
            return {
                "final_answer": "I was unable to retrieve relevant information for your question.",
                "error": "No evidence from any tool",
            }

        evidence = "\n\n---\n\n".join(evidence_parts)
        user_prompt = (
            f"User Question: {question}\n\n"
            f"Evidence:\n\n{evidence}\n\n"
            "Please synthesize a clear, complete answer based only on this evidence."
        )

        t0 = time.time()
        client = genai.Client(api_key=_api_key)
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYNTHESIZER_SYSTEM,
                        temperature=0.1,
                    ),
                )
                return {
                    "final_answer": response.text,
                    "metadata": {
                        **state.get("metadata", {}),
                        "synthesizer_ms": round((time.time() - t0) * 1000),
                    },
                }
            except Exception as exc:
                if "429" in str(exc):
                    if _is_daily_quota_exhausted(str(exc)):
                        return {
                            "final_answer": "Daily API quota exhausted. Please try again tomorrow.",
                            "error": str(exc),
                        }
                    if attempt < 2:
                        wait = 35 * (attempt + 1)
                        print(f"  [Synthesizer] Rate limited -- waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        return {
                            "final_answer": "An error occurred while generating the final answer.",
                            "error": str(exc),
                        }
        return {"final_answer": "Failed to generate answer after retries."}

    return synthesizer_node


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_agent_graph(conn, schema, collection, bm25, chunks, model: str = "gemini-3.1-flash-lite"):
    """
    Build and compile the LangGraph agent state graph.

    Parameters
    ----------
    conn       : sqlite3.Connection (open)
    schema     : dict from schema_utils.get_schema()
    collection : chromadb.Collection (from rag_pipeline.build_index)
    bm25       : BM25Okapi index (from rag_pipeline.build_index)
    chunks     : list[dict] of RAG chunks
    model      : Gemini model for planner and synthesizer

    Returns
    -------
    Compiled LangGraph runnable (.invoke(), .stream())

    Design: resources captured once in closures -- no per-call init cost.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("planner",     planner_node)
    graph.add_node("sql_tool",    make_sql_tool_node(conn, schema))
    graph.add_node("rag_tool",    make_rag_tool_node(collection, bm25, chunks))
    graph.add_node("both_tools",  make_both_tools_node(conn, schema, collection, bm25, chunks))
    graph.add_node("synthesizer", make_synthesizer_node(model=model))

    # Entry point
    graph.set_entry_point("planner")

    # Conditional routing from planner
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "SQL":      "sql_tool",
            "RAG":      "rag_tool",
            "BOTH":     "both_tools",
            "CHITCHAT": "synthesizer",
        },
    )

    # All tool nodes converge at synthesizer
    graph.add_edge("sql_tool",   "synthesizer")
    graph.add_edge("rag_tool",   "synthesizer")
    graph.add_edge("both_tools", "synthesizer")

    # Synthesizer -> END
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(question: str, graph) -> dict:
    """
    Run the agent on a single question.

    Returns
    -------
    {
        "question":     str,
        "route":        str,           # SQL | RAG | BOTH | CHITCHAT
        "route_reason": str,
        "final_answer": str,
        "sql_result":   dict | None,
        "rag_result":   dict | None,
        "error":        str | None,
        "metadata":     dict,
        "total_ms":     int,
    }
    """
    t0 = time.time()
    initial_state: AgentState = {
        "question":     question,
        "route":        None,
        "route_reason": None,
        "sql_result":   None,
        "rag_result":   None,
        "final_answer": None,
        "error":        None,
        "metadata":     {},
    }
    final_state = graph.invoke(initial_state)
    return {
        "question":     question,
        "route":        final_state.get("route",        "UNKNOWN"),
        "route_reason": final_state.get("route_reason", ""),
        "final_answer": final_state.get("final_answer", ""),
        "sql_result":   final_state.get("sql_result"),
        "rag_result":   final_state.get("rag_result"),
        "error":        final_state.get("error"),
        "metadata":     final_state.get("metadata", {}),
        "total_ms":     round((time.time() - t0) * 1000),
    }


# ---------------------------------------------------------------------------
# Startup helper -- load all resources once at startup
# ---------------------------------------------------------------------------

def load_resources(
    db_path:    Optional[Path] = None,
    docs_dir:   Optional[Path] = None,
    chroma_dir: str = "chroma_db",
):
    """
    Load the SQLite DB, schema, and RAG index.
    Designed to be called once before build_agent_graph().

    Returns (conn, schema, collection, bm25, chunks).
    """
    import sqlite3

    scripts_dir = str(Path(__file__).parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from schema_utils import get_schema
    from rag_pipeline import load_and_chunk_documents, build_index

    root     = Path(__file__).parent.parent
    db_path  = db_path  or (root / "db"   / "analytics.db")
    docs_dir = docs_dir or (root / "docs")

    print("Loading database...")
    conn   = sqlite3.connect(str(db_path))
    schema = get_schema(conn)
    total_rows = sum(v["row_count"] for v in schema.values())
    print(f"  {len(schema)} tables | {total_rows:,} total rows")

    print("Loading RAG index...")
    chunks = load_and_chunk_documents(docs_dir)
    collection, bm25, chunks = build_index(chunks, persist_dir=chroma_dir)
    print(f"  {len(chunks)} chunks indexed")

    return conn, schema, collection, bm25, chunks


# ---------------------------------------------------------------------------
# CLI entrypoint -- interactive single-question test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _api_key:
        sys.exit("GEMINI_API_KEY not set in .env")

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "What are the top 5 product categories by revenue, "
        "and what is the return policy for defective products?"
    )

    conn, schema, collection, bm25, chunks = load_resources()
    graph = build_agent_graph(conn, schema, collection, bm25, chunks)

    SEP = "=" * 70
    print(f"\n{SEP}")
    print(f"  Question : {question}")
    print(f"{SEP}\n")

    result = run_agent(question, graph)

    print(f"  Route    : {result['route']}")
    print(f"  Reason   : {result['route_reason']}")
    print(f"  Time     : {result['total_ms']}ms")

    if result["sql_result"]:
        s = result["sql_result"]
        status = "OK" if s.get("success") else "FAIL"
        print(f"  SQL      : [{status}] {s.get('row_count', 0)} rows "
              f"({s.get('tool_ms', 0)}ms, {s.get('attempts', 0)} attempt(s))")

    if result["rag_result"]:
        r = result["rag_result"]
        status = "OK" if r.get("answerable") else "FAIL"
        print(f"  RAG      : [{status}] answerable={r.get('answerable')} | "
              f"sources={r.get('sources', [])} ({r.get('tool_ms', 0)}ms)")

    print(f"\n{'-'*70}")
    print("Answer:\n")
    print(result["final_answer"])
    print(f"\n{SEP}")

    conn.close()
