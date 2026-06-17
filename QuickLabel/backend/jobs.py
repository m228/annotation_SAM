"""Background job manager for SAM operations.

SAM inference is slow (seconds to tens of seconds), so requests run as jobs on a
single worker thread. The UI starts a job, polls its progress, and can cancel
it. A single worker serializes all SAM work (there is only one model in memory),
while job status is read from a separate dict so polling never blocks on the
running inference.

Cancellation:
  * Multi-step jobs (propagation) check ``job.is_cancelled()`` between images and
    stop early — a true cancel.
  * A single torch inference cannot be interrupted mid-call; cancelling such a
    job stops the UI from waiting and discards the result when it finishes.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Optional


class Job:
    def __init__(self, job_type: str):
        self.id = uuid.uuid4().hex[:12]
        self.type = job_type
        self.status = "queued"          # queued | running | done | error | cancelled
        self.progress = {"percent": 0, "message": "В очереди…"}
        self.result: Any = None
        self.error: Optional[str] = None
        self.created = time.time()
        self._cancel = threading.Event()

    # — used by the worker function —
    def set_progress(self, percent: int, message: str) -> None:
        self.progress = {"percent": int(max(0, min(100, percent))), "message": message}

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "type": self.type, "status": self.status,
            "progress": self.progress, "result": self.result, "error": self.error,
        }


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[tuple[Job, Callable[[Job], Any]]]" = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, job_type: str, func: Callable[[Job], Any]) -> Job:
        self._prune()
        job = Job(job_type)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put((job, func))
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        job.cancel()
        if job.status == "queued":
            job.status = "cancelled"
        return True

    def _prune(self) -> None:
        """Drop finished jobs older than 10 minutes to bound memory."""
        cutoff = time.time() - 600
        with self._lock:
            stale = [jid for jid, j in self._jobs.items()
                     if j.status in ("done", "error", "cancelled") and j.created < cutoff]
            for jid in stale:
                del self._jobs[jid]

    def _run(self) -> None:
        while True:
            job, func = self._queue.get()
            if job.is_cancelled():
                job.status = "cancelled"
                continue
            job.status = "running"
            try:
                result = func(job)
                if job.is_cancelled():
                    job.status = "cancelled"
                else:
                    job.result = result
                    job.status = "done"
                    job.set_progress(100, "Готово")
            except Exception as exc:  # noqa: BLE001
                job.error = str(exc)
                job.status = "error"
                traceback.print_exc()


# Single shared manager for the server process.
manager = JobManager()
