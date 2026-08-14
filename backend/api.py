import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add scripts directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "scripts"))

from agent_router import load_resources, build_agent_graph, run_agent

app = FastAPI(title="DataTell API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryEntry(BaseModel):
    role: str    # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    conversation_history: List[HistoryEntry] = []
    # Phase 9 placeholder: route to a specific uploaded dataset
    dataset_id: str = "olist"


# Global state for graph
agent_graph = None

_db_path = root_dir / "db" / "analytics.db"


@app.on_event("startup")
def startup_event():
    global agent_graph
    print("Initializing DataTell Agent...")

    # Cloud mode flags — read from environment variables
    # Set USE_TURSO=true and CHROMA_MODE=cloud in Render env vars for production.
    use_turso   = os.getenv("USE_TURSO", "false").lower() == "true"
    chroma_mode = os.getenv("CHROMA_MODE", "local")

    print(f"  DB mode    : {'Turso Cloud' if use_turso else 'Local SQLite'}")
    print(f"  Chroma mode: {chroma_mode}")

    try:
        conn, schema, collection, bm25, chunks = load_resources(
            use_turso=use_turso,
            chroma_mode=chroma_mode,
        )

        if not use_turso:
            # FastAPI runs sync endpoints in worker threads, but conn was created in the startup thread.
            # This causes SQLite "objects created in a thread can only be used in that same thread" errors.
            # Since our queries are read-only, we can recreate the connection with check_same_thread=False.
            # (pyturso handles thread-safety internally, so this is only needed for local SQLite.)
            conn.close()
            thread_safe_conn = sqlite3.connect(str(_db_path), check_same_thread=False)
        else:
            thread_safe_conn = conn  # pyturso is thread-safe

        agent_graph = build_agent_graph(thread_safe_conn, schema, collection, bm25, chunks)
        print("Agent ready.")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")


def _serialize_result(result: dict) -> dict:
    """Convert DataFrame in sql_result to JSON-serializable dicts, replacing NaN with None."""
    sql_res = result.get("sql_result")
    if sql_res and sql_res.get("results") is not None:
        df = sql_res["results"].replace({np.nan: None})
        result["sql_result"]["results_json"] = df.to_dict(orient="records")
        del result["sql_result"]["results"]
    return result


def _run_agent_sync(question: str, history: list) -> dict:
    """Synchronous wrapper called from thread pool to avoid blocking the event loop."""
    result = run_agent(question, agent_graph, history=history)
    return _serialize_result(result)


@app.post("/query")
async def query(req: QueryRequest):
    """Standard JSON endpoint — waits for the full answer then returns it."""
    if not agent_graph:
        raise HTTPException(status_code=500, detail="Agent graph not initialized")
    try:
        history = [{"role": h.role, "content": h.content} for h in req.conversation_history]
        # Run blocking agent in a thread pool so we don't block the event loop,
        # with a 120-second hard timeout to prevent hung Gemini calls.
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, _run_agent_sync, req.question, history
            ),
            timeout=120.0,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Agent timed out after 120 seconds")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    SSE streaming endpoint.
    Emits events as the answer arrives so the frontend can typewriter-render them.

    Event protocol (each line: "data: <json>\\n\\n"):
      - {"type": "meta",   "route": "...", "route_reason": "..."}
      - {"type": "token",  "text": "<chunk>"}   (one per word / punctuation group)
      - {"type": "done",   "sql_result": ..., "rag_result": ..., "metadata": ...,
                           "answer_quality": ..., "critique": ..., "total_ms": ...}
      - {"type": "error",  "message": "..."}
    """
    if not agent_graph:
        raise HTTPException(status_code=500, detail="Agent graph not initialized")

    history = [{"role": h.role, "content": h.content} for h in req.conversation_history]

    async def event_generator():
        try:
            # Run the blocking agent in a thread pool
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _run_agent_sync, req.question, history),
                timeout=120.0,
            )

            # 1. Emit routing metadata immediately
            yield f"data: {json.dumps({'type': 'meta', 'route': result.get('route'), 'route_reason': result.get('route_reason')})}\n\n"

            # 2. Stream the final answer word by word for the typewriter effect
            answer = result.get("final_answer") or ""
            # Split into word-sized tokens (preserve spaces/newlines)
            import re
            tokens = re.findall(r'\S+|\s+', answer)
            for token in tokens:
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                # Small delay to create visible typewriter cadence
                await asyncio.sleep(0.012)

            # 3. Emit done event with all metadata
            done_payload = {
                "type": "done",
                "sql_result": result.get("sql_result"),
                "rag_result": result.get("rag_result"),
                "metadata": result.get("metadata", {}),
                "answer_quality": result.get("answer_quality"),
                "critique": result.get("critique"),
                "retry_count": result.get("retry_count", 0),
                "total_ms": result.get("total_ms"),
                "error": result.get("error"),
            }
            yield f"data: {json.dumps(done_payload)}\n\n"

        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Agent timed out after 120 seconds'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
        },
    )


if __name__ == "__main__":
    import uvicorn
    # Render sets the PORT env var; default to 8000 for local development
    port = int(os.getenv("PORT", 8000))
    # Determine import string depending on whether run from root or backend/
    app_target = "backend.api:app" if (root_dir / "backend").exists() and Path.cwd() == root_dir else "api:app"
    uvicorn.run(app_target, host="0.0.0.0", port=port, reload=(port == 8000))
