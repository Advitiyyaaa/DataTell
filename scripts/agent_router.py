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
import sqlite3
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
_api_key = os.getenv("GEMINI_API_KEY")

# Module-level Gemini client — instantiated once, reused across all nodes.
# None-safe: created lazily so import doesn't fail when API key is absent.
_client: Optional[genai.Client] = None

def _get_client() -> genai.Client:
    """Return the singleton Gemini client, creating it on first use."""
    global _client
    if _client is None:
        if not _api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Add it to .env or export it "
                "as an environment variable."
            )
        _client = genai.Client(api_key=_api_key)
    return _client

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Ensure scripts/ is importable for sibling module imports (nl_to_sql, rag_pipeline).
_scripts_dir = str(Path(__file__).parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from nl_to_sql import nl_to_sql          # noqa: E402  (after sys.path setup)
from rag_pipeline import rag_query       # noqa: E402

# ---------------------------------------------------------------------------
# Default model — single constant for easy switching across all nodes.
# Change this ONE value to swap models project-wide (quota fallback, upgrades).
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Mutable state passed between LangGraph nodes."""
    question:     str
    history:      Optional[list]  # [{"role": "user"|"assistant", "content": str}, ...]
    route:        Optional[str]   # "SQL" | "RAG" | "BOTH" | "CHITCHAT"
    route_reason: Optional[str]   # planner one-sentence reasoning
    sql_result:   Optional[dict]  # full dict from nl_to_sql()
    rag_result:   Optional[dict]  # full dict from rag_query()
    final_answer:   Optional[str]
    error:          Optional[str]
    metadata:       dict            # timing, model call info, etc.
    answer_quality: Optional[str]  # "PASS" | "FAIL"
    critique:       Optional[str]  # critic's feedback, injected into retry prompt
    retry_count:    int            # incremented by critic_node; max usable retries = 2


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
- RAG     : answerable purely from the policy documents (including hypothetical or procedural questions like "What happens if..." or "How do I...")
- BOTH    : requires BOTH database data AND policy document context
- CHITCHAT: greeting, meta-question about the system, or completely unanswerable

EXAMPLES:
  "What are the top 5 product categories by revenue?"                        -> SQL
  "How many orders were delivered late in 2017?"                             -> SQL
  "Which sellers have the highest average review score?"                     -> SQL
  "What is the return window for defective products?"                        -> RAG
  "What happens to a seller who receives low review scores?"                 -> RAG
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


def _is_daily_quota_exhausted(exc: Exception) -> bool:
    """Return True when the error is a daily (not per-minute) quota exhaustion."""
    msg = str(exc)
    return "GenerateRequestsPerDayPerProjectPerModel" in msg or "PerDay" in msg


def _classify_question(question: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Call Gemini to classify the question.
    Returns {"route": str, "reasoning": str}.

    Rate-limit handling:
    - Per-minute (429 without daily flag): retry with exponential backoff
    - Daily quota exhausted: raise immediately (backoff won't help)
    """
    client = _get_client()
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
        except ClientError as exc:
            if exc.code == 429:
                if _is_daily_quota_exhausted(exc):
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
        except Exception:
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

def make_sql_tool_node(
    conn, schema, model: str = DEFAULT_MODEL,
) -> Callable[[AgentState], dict]:
    """
    Factory capturing (conn, schema, model) in a closure.
    Returns a LangGraph-compatible node function.

    Generalisation note: `conn` and `schema` are injected at build time,
    so swapping the underlying database (Phase 9 "any CSV") only requires
    passing a different connection + schema — no changes to the graph.
    """
    def sql_tool_node(state: AgentState) -> dict:
        t0 = time.time()
        
        question = state["question"]

        # On retry: inject the critic's feedback so nl_to_sql writes a better query.
        if state.get("retry_count", 0) > 0 and state.get("critique"):
            question += (
                f"\n\nCRITIQUE OF PREVIOUS ATTEMPT:\n{state['critique']}\n\n"
                "Please rewrite your SQL to address ALL parts of the question above."
            )
        else:
            # On FIRST attempt: add a completeness hint to pre-empt the most common
            # Critic failure — the LLM writing a query that only covers one item
            # when the user asked for N. Rule 12 in the system prompt gives the CTE
            # pattern; this hint reminds the LLM to apply it.
            question += (
                "\n\n[IMPORTANT: If this question asks for details about the 'top N' items "
                "(e.g. reviews/prices for top 5 categories), use a CTE to first find all N items, "
                "then JOIN the detail table against that CTE so ALL N items are covered — "
                "never filter to just one hardcoded value.]"
            )

        try:
            result = nl_to_sql(
                question,
                schema,
                conn,
                model=model,
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

def make_rag_tool_node(
    collection, bm25, chunks, model: str = DEFAULT_MODEL,
) -> Callable[[AgentState], dict]:
    """
    Factory capturing (collection, bm25, chunks, model) in a closure.
    Returns a LangGraph-compatible node function.

    Generalisation note: the RAG index is injected at build time.
    Swapping the document corpus only requires rebuilding the index
    and passing the new (collection, bm25, chunks) triple.
    """
    def rag_tool_node(state: AgentState) -> dict:
        t0 = time.time()
        
        question = state["question"]
        if state.get("retry_count", 0) > 0 and state.get("critique"):
            question += f"\n\nCRITIQUE OF PREVIOUS ATTEMPT:\n{state['critique']}\n\nPlease address this critique."

        try:
            result = rag_query(
                question,
                collection,
                bm25,
                chunks,
                model=model,
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

def make_both_tools_node(
    conn, schema, collection, bm25, chunks,
    model: str = DEFAULT_MODEL,
    db_path: Optional[Path] = None,
) -> Callable[[AgentState], dict]:
    """
    Factory: runs SQL and RAG concurrently via ThreadPoolExecutor.

    Why threads (not async): nl_to_sql and rag_query make blocking
    API/DB calls. Threads let them run in parallel without rewriting them
    as coroutines. Wall-clock ~ max(sql_time, rag_time) not sql + rag.

    Thread-safety note: SQLite connections must NOT be shared across threads.
    We create a fresh, short-lived connection inside the SQL worker thread and
    close it when done. The RAG path (ChromaDB + BM25) is already thread-safe.
    """
    # Resolve DB path: use explicit arg, else infer from project layout
    _db_path = db_path or (Path(__file__).parent.parent / "db" / "analytics.db")
    _rag_fn  = make_rag_tool_node(collection, bm25, chunks, model=model)

    def _run_sql_in_thread(state: AgentState) -> dict:
        """Open a thread-local connection, run SQL, close it."""
        thread_conn = sqlite3.connect(str(_db_path))
        try:
            sql_fn = make_sql_tool_node(thread_conn, schema, model=model)
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

COMPLETENESS CHECK (do this BEFORE writing your response):
- Re-read the user's question and identify EVERY sub-part (e.g. if they asked for
  both "best" and "worst", you must address BOTH).
- If the evidence only covers SOME parts of the question, clearly state which parts
  are covered and which are missing — do NOT silently omit or hallucinate missing data.
- Never present a partial answer as if it were complete.
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


def make_synthesizer_node(model: str = DEFAULT_MODEL) -> Callable[[AgentState], dict]:
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
        client   = _get_client()
        history  = state.get("history") or []

        # Build a conversation-history prefix for multi-turn context
        def _history_block(history: list) -> str:
            if not history:
                return ""
            lines = []
            for turn in history[-6:]:  # cap at last 6 turns to stay within token budget
                role = "User" if turn.get("role") == "user" else "DataTell"
                lines.append(f"{role}: {turn.get('content', '').strip()}")
            return "Conversation so far:\n" + "\n".join(lines) + "\n\n"

        # CHITCHAT: conversational reply
        if route == "CHITCHAT":
            try:
                chitchat_prompt = _history_block(history) + question
                response = client.models.generate_content(
                    model=model,
                    contents=chitchat_prompt,
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
        history_prefix = _history_block(history)
        user_prompt = (
            f"{history_prefix}"
            f"User Question: {question}\n\n"
            f"Evidence:\n\n{evidence}\n\n"
            "Please synthesize a clear, complete answer based only on this evidence."
        )

        retry_count = state.get("retry_count", 0)
        if retry_count > 0 and state.get("critique"):
            user_prompt += (
                f"\n\n---\n\nPREVIOUS ANSWER (REJECTED):\n{state.get('final_answer')}\n\n"
                f"CRITIQUE: {state.get('critique')}\n\n"
                "Please rewrite your answer to address the critique above."
            )

        t0 = time.time()
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
                # Use a per-retry key so cumulative timing survives multiple passes.
                # retry_count is the count BEFORE this synthesizer call (incremented by critic).
                run_idx = state.get("retry_count", 0)
                timing_key = "synthesizer_ms" if run_idx == 0 else f"synthesizer_ms_retry{run_idx}"
                return {
                    "final_answer": response.text,
                    "metadata": {
                        **state.get("metadata", {}),
                        timing_key: round((time.time() - t0) * 1000),
                    },
                }
            except ClientError as exc:
                if exc.code == 429:
                    if _is_daily_quota_exhausted(exc):
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
                else:
                    return {
                        "final_answer": "An error occurred while generating the final answer.",
                        "error": str(exc),
                    }
            except Exception as exc:
                return {
                    "final_answer": "An error occurred while generating the final answer.",
                    "error": str(exc),
                }
        return {"final_answer": "Failed to generate answer after retries."}

    return synthesizer_node


# ---------------------------------------------------------------------------
# Critic Node -- Self-Check loop for hallucinations
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = """\
You are an expert evaluator for the DataTell analytics assistant.
Your job is to evaluate if the generated evidence (SQL data or RAG text) is SUFFICIENT to fully answer the user's question, AND if the assistant's final answer is grounded in that evidence.

CRITERIA:
1. EVIDENCE COMPLETENESS: Does the provided evidence fully cover ALL parts of the user's question? If the user asks for multiple things (e.g., "best and worst"), and the evidence only contains one (e.g., only "best"), you MUST FAIL it.
   IMPORTANT: The evidence shown to you is a SAMPLE of the full SQL result. The header tells you the TOTAL row count and which unique categories/groups are present. If the header says N total rows and lists multiple distinct categories, the evidence IS complete even if only a few rows per category are shown.
2. DIRECTNESS: Is the final answer directly addressing the user's question?
3. GROUNDEDNESS: Is the answer strictly derived from the provided evidence? (No fabrication/hallucination).
   ALLOWED TRANSFORMATIONS (these are NOT hallucination):
   - Translating text from Portuguese (or any language) to English when the user asked for English
   - Reformatting numbers, dates, or currency into readable form
   - Summarising or paraphrasing long text present in the evidence
   - Computing simple derived values (e.g. percentage from count/total both in evidence)
   REAL HALLUCINATION examples (these ARE fabrication):
   - Citing a category name that does not appear in the SQL result at all
   - Inventing a review score or review text not present in the evidence
   - Adding extra rows beyond what the SQL returned

Respond with ONLY valid JSON (no markdown fences or preamble):
{"quality": "PASS", "critique": "Looks good. Evidence fully covers the request."}
OR
{"quality": "FAIL", "critique": "<Specific reason why it failed. E.g., 'The SQL evidence only contains best reviews, but the user also asked for worst.'>"}
"""

def make_critic_node(model: str = DEFAULT_MODEL) -> Callable[[AgentState], dict]:
    def critic_node(state: AgentState) -> dict:
        route = state.get("route", "CHITCHAT")
        # CHITCHAT routes bypass the critic: no factual evidence to check groundedness against.
        # retry_count is not incremented so the value from run_agent's initial state is preserved.
        if route == "CHITCHAT":
            return {
                "answer_quality": "PASS",
                "critique": "Bypassed for CHITCHAT — no evidence grounding required.",
                "retry_count": state.get("retry_count", 0),
            }

        question = state["question"]
        final_answer = state.get("final_answer", "")
        
        sql_result = state.get("sql_result")
        rag_result = state.get("rag_result")

        # Build a COMPACT evidence summary for the critic.
        # Sending the full SQL table (potentially hundreds of rows) wastes tokens;
        # the critic only needs enough context to verify groundedness.
        evidence_parts = []
        if sql_result is not None:
            if sql_result.get("success") and sql_result.get("results") is not None:
                df_full = sql_result["results"]
                row_count = sql_result.get("row_count", len(df_full))

                # Stratified sample: show up to 2 rows per unique value of the first
                # column so the critic can see ALL groups are present (e.g. all 5
                # categories), not just the first 5 rows which may all be one group.
                first_col = df_full.columns[0] if len(df_full.columns) > 0 else None
                if first_col is not None and df_full[first_col].nunique() > 1:
                    df_sample = (
                        df_full.groupby(first_col, sort=False)
                        .head(2)
                        .head(20)  # cap total at 20 rows
                    )
                    unique_vals = df_full[first_col].unique().tolist()
                    unique_summary = f"Distinct '{first_col}' values ({len(unique_vals)}): {unique_vals}"
                else:
                    df_sample = df_full.head(10)
                    unique_summary = ""

                sample_str = df_sample.to_string(index=False)
                evidence_parts.append(
                    f"[SQL — {row_count} total rows, stratified sample shown]\n"
                    + (f"{unique_summary}\n" if unique_summary else "")
                    + sample_str
                )
            else:
                evidence_parts.append(_format_sql_for_synthesis(sql_result))
        if rag_result is not None:
            evidence_parts.append(_format_rag_for_synthesis(rag_result))

        if not evidence_parts:
            evidence = "[No evidence provided]"
        else:
            evidence = "\n\n---\n\n".join(evidence_parts)
             
        user_prompt = (
            f"User Question: {question}\n\n"
            f"Evidence:\n\n{evidence}\n\n"
            f"Assistant Answer:\n{final_answer}\n\n"
            "Evaluate the answer based on the criteria."
        )

        client = _get_client()
        t0 = time.time()
        
        current_retry = state.get("retry_count", 0)

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_CRITIC_SYSTEM,
                        temperature=0.0,
                    ),
                )
                text = response.text.strip()
                if text.startswith("```"):
                    inner = text.split("```")
                    text = inner[1] if len(inner) > 1 else inner[0]
                    if text.lower().startswith("json"):
                        text = text[4:].strip()
                
                result = json.loads(text.strip())
                quality = result.get("quality", "FAIL").upper()
                if quality not in ["PASS", "FAIL"]:
                    quality = "FAIL"
                
                output = {
                    "answer_quality": quality,
                    "critique": result.get("critique", ""),
                    "retry_count": current_retry + 1,
                    "metadata": {
                        **state.get("metadata", {}),
                        "critic_ms": state.get("metadata", {}).get("critic_ms", 0) + round((time.time() - t0) * 1000),
                    }
                }
                
                # If retries are exhausted and still failing, overwrite the hallucinated answer
                if quality == "FAIL" and current_retry >= 2:
                    output["final_answer"] = (
                        "I apologize, but after multiple attempts, I could not synthesize a completely accurate and grounded answer based on the available data.\n\n"
                        f"**Self-Evaluation Issue:** {result.get('critique', '')}\n\n"
                        "Please try rephrasing your question or narrowing your request."
                    )
                    
                return output
            except json.JSONDecodeError:
                # Retry silently; if all 3 attempts fail, the post-loop return handles it.
                pass
            except ClientError as exc:
                if exc.code == 429:
                    if _is_daily_quota_exhausted(exc):
                        return {"answer_quality": "PASS", "critique": "Quota exhausted during critique, passed to avoid loop", "retry_count": current_retry}
                    if attempt < 2:
                        time.sleep(35 * (attempt + 1))
                    else:
                        return {"answer_quality": "PASS", "critique": "Rate limit exhausted, passed to avoid loop", "retry_count": current_retry}
                else:
                    return {"answer_quality": "PASS", "critique": f"Error: {exc}, passed to avoid loop", "retry_count": current_retry}
            except Exception as exc:
                return {"answer_quality": "PASS", "critique": f"Error: {exc}, passed to avoid loop", "retry_count": current_retry}

        # All 3 LLM attempts returned malformed JSON — return FAIL so the
        # retry loop can attempt a fresh synthesis rather than silently passing.
        output = {
            "answer_quality": "FAIL",
            "critique": "Critic could not parse evaluator response after 3 attempts.",
            "retry_count": current_retry + 1,
        }
        if current_retry >= 2:
            output["final_answer"] = (
                "I apologize, but I encountered an internal error during self-evaluation. "
                "I cannot guarantee the accuracy of the current response, so it has been withheld."
            )
        return output
    
    return critic_node


# ---------------------------------------------------------------------------
# Critic routing helper -- module-level so it is unit-testable in isolation
# ---------------------------------------------------------------------------

def route_after_critic(state: AgentState) -> str:
    """
    Conditional edge from critic:
      PASS            -> END
      FAIL + retries  -> route back to the tool node (max 2 retries)
      FAIL + exhausted-> END (pass through best available answer)
    """
    if state.get("answer_quality", "PASS") == "PASS":
        return "END"
    
    if state.get("retry_count", 0) < 3:
        route = state.get("route", "CHITCHAT")
        if route == "SQL":
            return "sql_tool"
        elif route == "RAG":
            return "rag_tool"
        elif route == "BOTH":
            return "both_tools"
        return "synthesizer" # fallback
        
    return "END"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_agent_graph(
    conn, schema, collection, bm25, chunks,
    model: str = DEFAULT_MODEL,
):
    """
    Build and compile the LangGraph agent state graph.

    Parameters
    ----------
    conn       : sqlite3.Connection (open)
    schema     : dict from schema_utils.get_schema()
    collection : chromadb.Collection (from rag_pipeline.build_index)
    bm25       : BM25Okapi index (from rag_pipeline.build_index)
    chunks     : list[dict] of RAG chunks
    model      : Gemini model for ALL nodes (planner, tools, synthesizer, critic)

    Returns
    -------
    Compiled LangGraph runnable (.invoke(), .stream())

    Design
    ------
    Resources are captured once in closures — no per-call init cost.
    The `model` param flows to every node, ensuring a single point of
    control for model selection (makes quota-fallback a one-line change).

    Graph topology (Phase 5+):
        planner --(route)--> sql_tool  --+
                         --> rag_tool  --+--> synthesizer --> critic --[PASS]--> END
                         --> both_tools--+                       |
                         --> synthesizer (CHITCHAT)              +--[FAIL, retries<2]--> synthesizer

    Generalisation (Phase 9 "any CSV"): only `conn`, `schema`, and the
    RAG triple need to change — the graph topology stays the same.
    """
    # Validate early — fail here rather than deep inside a node
    _get_client()  # ensures API key is present

    graph = StateGraph(AgentState)

    # Register nodes — model param flows through to all factories
    graph.add_node("planner",     planner_node)
    graph.add_node("sql_tool",    make_sql_tool_node(conn, schema, model=model))
    graph.add_node("rag_tool",    make_rag_tool_node(collection, bm25, chunks, model=model))
    graph.add_node("both_tools",  make_both_tools_node(
        conn, schema, collection, bm25, chunks, model=model,
    ))
    graph.add_node("synthesizer", make_synthesizer_node(model=model))
    graph.add_node("critic",      make_critic_node(model=model))

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

    # Synthesizer -> Critic
    graph.add_edge("synthesizer", "critic")

    # Critic -> conditional edge (route_after_critic is module-level for testability)
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "END": END,
            "sql_tool": "sql_tool",
            "rag_tool": "rag_tool",
            "both_tools": "both_tools",
            "synthesizer": "synthesizer",
        }
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(question: str, graph, history: Optional[list] = None) -> dict:
    """
    Run the agent on a single question.

    Parameters
    ----------
    question : str
    graph    : compiled LangGraph
    history  : optional list of {"role": str, "content": str} dicts representing
               prior conversation turns (most recent last). Capped to last 6 turns
               inside the synthesizer to stay within token budget.

    Returns
    -------
    {
        "question":      str,
        "route":         str,           # SQL | RAG | BOTH | CHITCHAT
        "route_reason":  str,
        "final_answer":  str,
        "sql_result":    dict | None,
        "rag_result":    dict | None,
        "error":         str | None,
        "metadata":      dict,          # timing per node (synthesizer_ms, critic_ms, ...)
        "total_ms":      int,
        "answer_quality": str | None,  # "PASS" | "FAIL" from critic
        "critique":      str | None,   # critic's feedback (present when FAIL or CHITCHAT bypass)
        "retry_count":   int,          # number of critic evaluations performed (0 for CHITCHAT)
    }
    """
    t0 = time.time()
    initial_state: AgentState = {
        "question":     question,
        "history":      history or [],
        "route":        None,
        "route_reason": None,
        "sql_result":   None,
        "rag_result":   None,
        "final_answer": None,
        "error":        None,
        "metadata":     {},
        "answer_quality": None,
        "critique":     None,
        "retry_count":  0,
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
        "answer_quality": final_state.get("answer_quality"),
        "critique":     final_state.get("critique"),
        "retry_count":  final_state.get("retry_count", 0),
    }


# ---------------------------------------------------------------------------
# Startup helper -- load all resources once at startup
# ---------------------------------------------------------------------------

def load_resources(
    db_path:    Optional[Path] = None,
    docs_dir:   Optional[Path] = None,
    chroma_dir: Optional[Path] = None,
):
    """
    Load the SQLite DB, schema, and RAG index.
    Designed to be called once before build_agent_graph().

    Generalisation note (Phase 9 "any CSV"): all three resource paths
    are parameterised.  Callers can point at a user-uploaded DB and a
    different docs directory without touching any other code.

    Returns (conn, schema, collection, bm25, chunks).
    """
    from schema_utils import get_schema
    from rag_pipeline import load_and_chunk_documents, build_index

    root       = Path(__file__).parent.parent
    db_path    = db_path    or (root / "db"   / "analytics.db")
    docs_dir   = docs_dir   or (root / "docs")
    chroma_dir = chroma_dir or (root / "chroma_db")

    print("Loading database...")
    conn   = sqlite3.connect(str(db_path))
    schema = get_schema(conn)
    total_rows = sum(v["row_count"] for v in schema.values())
    print(f"  {len(schema)} tables | {total_rows:,} total rows")

    print("Loading RAG index...")
    chunks = load_and_chunk_documents(docs_dir)
    collection, bm25, chunks = build_index(chunks, persist_dir=str(chroma_dir))
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

    if result.get("answer_quality"):
        print(f"  Critic   : [{result['answer_quality']}] retries={result['retry_count']}")
        if result.get("critique") and result["answer_quality"] != "PASS":
            print(f"  Critique : {result['critique']}")

    print(f"\n{'-'*70}")
    print("Answer:\n")
    print(result["final_answer"])
    print(f"\n{SEP}")

    conn.close()
