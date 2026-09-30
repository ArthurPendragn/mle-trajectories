"use client";

import { useState } from "react";
import { stopJobAction } from "@/app/actions/run-actions";
import type { Job } from "@/lib/types";

export const live = (j: Job) => j.state === "running" || j.state === "starting";

function fmtTime(iso: string | null): string {
  return iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" }) : "";
}

/** One job: state, progress (sweeps), log, stop, and a link to what it made. */
export function JobCard({ job, stores, onChange }: {
  job: Job;
  stores: string[];              // runtime stores the page knows (for result links)
  onChange: (job: Job) => void;
}) {
  const [confirmStop, setConfirmStop] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const p = job.progress;
  const store = job.outputs.runtime;

  async function stop() {
    const res = await stopJobAction(job.id);
    if (res.ok) onChange(res.value);
    else setError(res.error);
    setConfirmStop(false);
  }

  return (
    <div className={`job job-${job.state}`}>
      <div className="job-head">
        <b>{job.label}</b>
        <span className={`badge state-${job.state}`}>
          {live(job) && <span className="spinner" />} {job.state}
        </span>
        <span className="muted small">
          started {fmtTime(job.started_at)}
          {job.finished_at && ` · ended ${fmtTime(job.finished_at)}`}
          {job.state === "failed" && job.returncode != null && ` · exit ${job.returncode}`}
        </span>
        <span className="job-links">
          {store && stores.includes(store) && job.params.source && (
            <a href={`?source=${encodeURIComponent(job.params.source)}&runtime=${encodeURIComponent(store)}#runtime-profile`}>
              show in runtime profile
            </a>
          )}
          {live(job) && (confirmStop
            ? <>
                <button onClick={stop}>stop now</button>
                <button onClick={() => setConfirmStop(false)}>keep running</button>
              </>
            : <button onClick={() => setConfirmStop(true)}
                      title={job.action === "runtime-sweep"
                        ? "stops the current pipeline; everything measured so far stays in the store"
                        : "stops the build; the existing sample, if any, is left as it was"}>
                stop
              </button>)}
        </span>
      </div>
      {job.action === "runtime-sweep" && p.todo != null && (
        <div className="job-progress small">
          <progress value={p.done} max={Math.max(p.todo, 1)} />
          <span>
            {p.done}/{p.todo} measured
            {p.failed > 0 && <span className="down"> · {p.failed} of them failed</span>}
            {p.cached ? <span className="muted"> · {p.cached} already fresh</span> : null}
            {live(job) && p.current && <span className="muted"> · now <span className="mono">{p.current}</span></span>}
          </span>
        </div>
      )}
      {error && <p className="error small">{error}</p>}
      <details className="block" open={job.state === "failed" || job.state === "lost"
                                        || (job.action === "build-sample" && live(job))}>
        <summary>log</summary>
        <pre className="log">{job.log || "(empty)"}</pre>
      </details>
    </div>
  );
}
