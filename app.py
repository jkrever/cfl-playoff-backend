"""
CFL Playoff Scenario API -- a thin job-queue wrapper around the
already-validated cfl_engine.py. Design choices, and why:

- Job-submit + poll, not a single long request. Some checks in this
  project have needed tens of thousands of search candidates and
  several minutes to reach a real proof (not a guess) -- a plain
  request/response call risks getting killed by a host's request
  timeout mid-search. Submitting a job and polling for its result
  sidesteps that entirely, on any host, free or paid.

- In-memory job store, single background thread per job. This is a
  low-traffic personal tool, not a multi-user production service --
  no database, no task queue infrastructure needed. The real cost is
  the CP-SAT solving itself, not job bookkeeping. Jobs are lost if the
  server restarts (e.g. Render's free tier sleeping after inactivity),
  which is an acceptable tradeoff for this use case: if that happens,
  the user just submits again.

- The actual computation (cfl_engine.run_full_scan) is completely
  unchanged from the validated Colab script -- this file only adds the
  web plumbing around it.
"""
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import cfl_engine

app = FastAPI(title="CFL Playoff Scenario API")

# Allow the frontend (served from a different origin) to call this API.
# Tighten allow_origins to your actual frontend's domain once deployed --
# "*" is fine for local development and initial testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- In-memory job store ----
# {job_id: {"status": "running"|"done"|"error", "result": ..., "error": ...,
#           "created_at": <float>, "finished_at": <float or None>}}
_jobs = {}
_jobs_lock = threading.Lock()


def _run_scan_job(job_id: str, season_id: Optional[int], year: Optional[int]):
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        # verbose=False: the web job doesn't need the full deep-dive
        # progress log that the Colab script prints -- only the final
        # structured result matters here. Nothing about the underlying
        # search logic changes; this only silences its own print calls.
        result = cfl_engine.run_full_scan(season_id=season_id, year=year,
                                           send_email=False, verbose=False)
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["finished_at"] = time.time()
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["finished_at"] = time.time()


@app.post("/api/scan")
def start_scan(season_id: Optional[int] = None, year: Optional[int] = None):
    """
    Kicks off a fresh full scan in the background and returns immediately
    with a job_id. Poll GET /api/scan/{job_id} for the result.
    season_id/year are optional query params (e.g. POST /api/scan?year=2027)
    -- omit both to use cfl_engine's current-season defaults.
    """
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "result": None, "error": None,
            "created_at": time.time(), "finished_at": None,
        }
    thread = threading.Thread(target=_run_scan_job, args=(job_id, season_id, year), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued"}



@app.get("/api/scan/{job_id}")
def get_scan_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id (may have been lost on a server restart)")
    # Copy out just what the caller needs -- don't leak the lock/thread internals.
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job["result"],
        "error": job["error"],
        "created_at": job["created_at"],
        "finished_at": job["finished_at"],
    }


@app.get("/api/health")
def health():
    """Cheap endpoint for the frontend (or a keep-alive ping) to check
    the server is awake, without triggering an actual scan."""
    return {"status": "ok"}
