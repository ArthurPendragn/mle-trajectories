"use client";

import { Fragment, useEffect, useState } from "react";
import Link from "next/link";
import { stopJobAction } from "@/app/actions/run-actions";
import { fmtSecs } from "@/lib/format";
import type { Job, JobList } from "@/lib/types";
import { live } from "./job-card";

const POLL_MS = 3000;
const ACTIONS: Record<string, string> = { "runtime-sweep": "runtime sweep", "build-sample": "build sample" };

function fmtTime(iso: string | null): string {
  return iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" }) : "";
}

function duration(job: Job, now: number): string {
  const start = Date.parse(job.started_at);
  const end = job.finished_at ? Date.parse(job.finished_at) : live(job) ? now : NaN;
  return Number.isFinite(end - start) ? fmtSecs((end - start) / 1000) : "";
}

function Progress({ job }: { job: Job }) {
  const p = job.progress;
  if (job.action !== "runtime-sweep" || p.todo == null) return null;
  return (
    <span className="small">
      <progress value={p.done} max={Math.max(p.todo, 1)} /> {p.done}/{p.todo}
      {p.failed > 0 && <span className="down"> · {p.failed} failed</span>}
      {live(job) && p.current && <span className="muted"> · <span className="mono">{p.current}</span></span>}
    </span>
  );
}

function StopButton({ job, onChange }: { job: Job; onChange: (j: Job) => void }) {
  const [confirm, setConfirm] = useState(false);
  if (!live(job)) return null;
  async function stop() {
    const res = await stopJobAction(job.id);
    if (res.ok) onChange(res.value);
    setConfirm(false);
  }
  return confirm
    ? <><button onClick={stop}>stop now</button> <button onClick={() => setConfirm(false)}>keep</button></>
    : <button onClick={() => setConfirm(true)}>stop</button>;
}

/** All jobs on the node, filterable, with their log one click away. */
export function JobsOverview({ initial }: { initial: JobList }) {
  const [list, setList] = useState(initial);
  const [state, setState] = useState<"all" | "live" | "ended">("all");
  const [action, setAction] = useState("all");
  const [open, setOpen] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const anyLive = list.jobs.some(live);

  useEffect(() => {
    if (!anyLive) return;
    const t = setTimeout(async () => {
      setNow(Date.now());
      try {
        const res = await fetch("/api/jobs", { cache: "no-store" });
        if (res.ok) setList(await res.json());
      } catch {
        /* next tick retries */
      }
    }, POLL_MS);
    return () => clearTimeout(t);
  }, [list, anyLive]);

  function upsert(job: Job) {
    setList((cur) => ({ ...cur, jobs: cur.jobs.map((j) => (j.id === job.id ? job : j)) }));
  }

  const shown = list.jobs.filter((j) =>
    (state === "all" || (state === "live") === live(j)) && (action === "all" || j.action === action));

  if (list.jobs.length === 0) return <p className="muted">No jobs yet.</p>;

  return (
    <>
      <div className="tree-modes">
        <label>show
          <select value={state} onChange={(e) => setState(e.target.value as typeof state)}>
            <option value="all">all ({list.jobs.length})</option>
            <option value="live">running ({list.jobs.filter(live).length})</option>
            <option value="ended">ended</option>
          </select>
        </label>
        <label>action
          <select value={action} onChange={(e) => setAction(e.target.value)}>
            <option value="all">all</option>
            {Object.entries(ACTIONS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </label>
        <span>click a row for its log</span>
      </div>
      <table className="jobs">
        <thead>
          <tr>
            <th>Started</th><th>Job</th><th>Run</th><th>State</th><th>Progress</th>
            <th className="num">Duration</th><th />
          </tr>
        </thead>
        <tbody>
          {shown.map((j) => {
            const [ds, run] = (j.run ?? "").split("/");
            return (
              <Fragment key={j.id}>
                <tr className="job-row" onClick={() => setOpen(open === j.id ? null : j.id)}>
                  <td className="small">{fmtTime(j.started_at)}</td>
                  <td>
                    {j.label}
                    <div className="muted small mono">{j.id}</div>
                  </td>
                  <td className="small">
                    {ds && run
                      ? <Link href={`/runs/${ds}/${run}#actions`} onClick={(e) => e.stopPropagation()}>{j.run}</Link>
                      : j.run}
                  </td>
                  <td>
                    <span className={`badge state-${j.state}`}>
                      {live(j) && <span className="spinner" />} {j.state}
                    </span>
                  </td>
                  <td><Progress job={j} /></td>
                  <td className="num small">{duration(j, now)}</td>
                  <td onClick={(e) => e.stopPropagation()}><StopButton job={j} onChange={upsert} /></td>
                </tr>
                {open === j.id && (
                  <tr className="job-log">
                    <td colSpan={7}>
                      <p className="mono muted small">{j.command}</p>
                      <pre className="log">{j.log || "(empty)"}</pre>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </>
  );
}
