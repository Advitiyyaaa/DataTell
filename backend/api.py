import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

class QueryRequest(BaseModel):
    question: str

# Global state for graph
agent_graph = None

@app.on_event("startup")
def startup_event():
    global agent_graph
    print("Initializing DataTell Agent...")
    try:
        conn, schema, collection, bm25, chunks = load_resources()
        
        # FastAPI runs sync endpoints in worker threads, but conn was created in the startup thread.
        # This causes SQLite "objects created in a thread can only be used in that same thread" errors.
        # Since our queries are read-only, we can recreate the connection with check_same_thread=False.
        conn.close()
        import sqlite3
        db_path = root_dir / "db" / "analytics.db"
        thread_safe_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        agent_graph = build_agent_graph(thread_safe_conn, schema, collection, bm25, chunks)
        print("Agent ready.")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")

@app.post("/query")
def query(req: QueryRequest):
    if not agent_graph:
        raise HTTPException(status_code=500, detail="Agent graph not initialized")
    try:
        result = run_agent(req.question, agent_graph)
        # Convert pandas DataFrames in sql_result to lists of dicts so they can be JSON serialized
        if result.get("sql_result") and result["sql_result"].get("results") is not None:
            df = result["sql_result"]["results"]
            # Convert NaN to None for JSON
            import numpy as np
            df = df.replace({np.nan: None})
            result["sql_result"]["results_json"] = df.to_dict(orient="records")
            # Don't delete the dataframe entirely in case it breaks something internally, 
            # but FastAPI won't be able to serialize a DataFrame directly.
            # We'll pop it out of the returned dict.
            del result["sql_result"]["results"]
            
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
