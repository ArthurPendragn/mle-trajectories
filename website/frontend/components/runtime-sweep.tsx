"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { planSweepAction, startSweepAction, stopJobAction } from "@/app/actions/sweep";
import type { ActionsInfo, Job, SweepParams, SweepPlan } from "@/lib/types";

const POLL_MS = 3000;
const CAPS = ["", "10000", "100000", "1000000"];

const live = (j: Job) => j.state === "running" || j.state === "starting";

function fmtTime(iso: string | null): string {
  return iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" }) : "";
}

function useActions(dataset: string, run: string, initial: ActionsInfo) {
  const router = useRouter();
  const [info, setInfo] = useState(initial);
  useEffect(() => setInfo(initial), [initial]);
  const wasLive = useRef(initial.jobs.some(live));
  const anyLive = info.jobs.some(live);

  useEffect(() => {
    // a job just ended: re-render the page so its runtime store shows up
    if (wasLive.current && !anyLive) router.refresh();
    wasLive.current = anyLive;
    if (!anyLive) return;
    const t = setTimeout(async () => {
      try {
        const res = await fetch(`/api/actions/${encodeURIComponent(dataset)}/${encodeURIComponent(run)}`,
                                { cache: "no-store" });
        if (res.ok) setInfo(await res.json());
      } catch {
        /* next tick retries */
      }
    }, POLL_MS);
    return () => clearTimeout(t);
  }, [info, anyLive, dataset, run, router]);

  return { info, setInfo, refresh: router.refresh };
}

// --------------------------------------------------------------------------- //
function SweepForm({ dataset, run, info, selectedSource, onStarted }: {
  dataset: string;
  run: string;
  info: ActionsInfo;
  selectedSource: string | null;
  onStarted: (job: Job) => void;
}) {
  const { sources, data } = info.sweep;
  const [source, setSource] = useState(
    sources.find((s) => s.name === selectedSource)?.name ?? sources.find((s) => s.default)?.name
    ?? sources[0]?.name ?? "");
  const [dataName, setDataName] = useState(data.find((d) => d.ok)?.name ?? "input");
  const [cap, setCap] = useState("");               // "" none, a preset, or "custom"
  const [customCap, setCustomCap] = useState("");
  const [timeoutMin, setTimeoutMin] = useState("60");
  const [retryFailed, setRetryFailed] = useState(false);
  const [force, setForce] = useState(false);

  const [plan, setPlan] = useState<SweepPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [listing, setListing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [starting, setStarting] = useState(false);

  const rows = cap === "custom" ? Number(customCap.replaceAll(/[,_\s]/g, "")) || null
    : cap ? Number(cap) : null;
  const params: SweepParams = {
    source, data: dataName, sample_rows: rows,
    timeout_s: Math.round((Number(timeoutMin) || 60) * 60), retry_failed: retryFailed, force,
  };
  const key = JSON.stringify(params);

  // what these parameters would write (instant; --list only on request)
  useEffect(() => {
    let stale = false;
    setConfirming(false);
    const t = setTimeout(async () => {
      const res = await planSweepAction(dataset, run, JSON.parse(key), false);
      if (stale) return;
      setPlan(res.ok ? res.value : null);
      setError(res.ok ? null : res.error);
    }, 250);
    return () => {
      stale = true;
      clearTimeout(t);
    };
  }, [dataset, run, key]);

  async function check() {
    setListing(true);
    try {
      const res = await planSweepAction(dataset, run, params, true);
      if (res.ok) setPlan(res.value);
      else setError(res.error);
    } finally {
      setListing(false);
    }
  }

  async function start() {
    setStarting(true);
    try {
      const res = await startSweepAction(dataset, run, params);
      if (res.ok) {
        setConfirming(false);
        onStarted(res.value);
      } else {
        setError(res.error);
      }
    } finally {
      setStarting(false);
    }
  }

  const dataOpt = data.find((d) => d.name === dataName);
  const count = plan?.listing?.would_run ?? null;

  return (
    <div className="sweep">
      <div className="sweep-form">
        <label>
          <span className="muted">skrub plans</span>
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            {sources.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name} ({s.coverage[0]}/{s.coverage[1]}){s.label ? ` — ${s.label}` : ""}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span className="muted">data</span>
          <select value={dataName} onChange={(e) => setDataName(e.target.value)}>
            {data.map((d) => (
              <option key={d.name} value={d.name} disabled={!d.ok}>
                {d.label}{d.ok ? "" : " — unavailable"}
              </option>
            ))}
          </select>
        </label>
        <label title="--sample-rows: every pandas.read_csv reads at most this many rows (parquet and other readers are not capped)">
          <span className="muted">row cap</span>
          <span className="sweep-cap">
            <select value={cap} onChange={(e) => setCap(e.target.value)}>
              {CAPS.map((c) => <option key={c} value={c}>{c ? Number(c).toLocaleString() : "none"}</option>)}
              <option value="custom">other…</option>
            </select>
            {cap === "custom" && (
              <input inputMode="numeric" placeholder="rows" value={customCap}
                     onChange={(e) => setCustomCap(e.target.value)} />
            )}
          </span>
        </label>
        <label>
          <span className="muted">timeout per pipeline</span>
          <span className="sweep-cap">
            <input inputMode="numeric" value={timeoutMin} onChange={(e) => setTimeoutMin(e.target.value)} /> min
          </span>
        </label>
        <label className="check" title="re-measure entries whose last run failed or timed out">
          <input type="checkbox" checked={retryFailed} onChange={(e) => setRetryFailed(e.target.checked)} />
          retry failed
        </label>
        <label className="check" title="re-measure every pipeline, including fresh entries">
          <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
          re-measure all
        </label>
      </div>
      {dataOpt?.note && <p className="muted small">{dataOpt.note}</p>}

      {error && <p className="error small">{error}</p>}
      {plan && !error && (
        <div className="small">
          <p>
            Writes <span className="mono">{plan.store}</span>{" "}
            {plan.store_summary ? (
              <span className="muted">
                — continues that store ({plan.store_summary.n_ok} ok
                {plan.store_summary.n_stale > 0 && `, ${plan.store_summary.n_stale} stale`}
                {plan.store_summary.n_failed > 0 && `, ${plan.store_summary.n_failed} failed`}):
                fresh entries are kept, stale and missing ones re-measured
              </span>
            ) : <span className="muted">— a new store</span>}
            .{" "}
            {plan.listing?.would_run != null
              ? <b>Would measure {plan.listing.would_run} of {plan.listing.total} pipelines.</b>
              : <span className="muted">{plan.n_pipelines} pipelines in scope.</span>}
          </p>
          <p className="mono muted">{plan.command}</p>
          {plan.listing && (
            <details className="block">
              <summary>what the tool would run (--list)</summary>
              <pre className="log">{plan.listing.text}</pre>
            </details>
          )}
        </div>
      )}

      <div className="sweep-buttons">
        <button onClick={check} disabled={!plan || !!error || listing}>
          {listing ? <><span className="spinner" /> checking…</> : "check what would run"}
        </button>
        {info.busy ? (
          <span className="muted small">
            A sweep is running ({info.busy.run}); one at a time, since sweeps would slow each other down.
          </span>
        ) : !confirming ? (
          <button className="primary" onClick={() => setConfirming(true)} disabled={!plan || !!error}>
            start sweep…
          </button>
        ) : (
          <>
            <span className="small">
              Measure {count ?? `up to ${plan?.n_pipelines}`} pipeline(s) on{" "}
              {dataOpt?.label}{rows ? `, ${rows.toLocaleString()} rows` : ""}, each for at most{" "}
              {timeoutMin} min?
            </span>
            <button className="primary" onClick={start} disabled={starting}>
              {starting ? "starting…" : "start"}
            </button>
            <button onClick={() => setConfirming(false)}>cancel</button>
          </>
        )}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- //
function JobCard({ job, stores, onChange }: {
  job: Job;
  stores: string[];
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
          {store && stores.includes(store) && (
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
                      title="stops the current pipeline; everything measured so far stays in the store">
                stop
              </button>)}
        </span>
      </div>
      {p.todo != null && (
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
      <details className="block" open={job.state === "failed" || job.state === "lost"}>
        <summary>log</summary>
        <pre className="log">{job.log || "(empty)"}</pre>
      </details>
    </div>
  );
}

// --------------------------------------------------------------------------- //
export function RuntimeSweep({ dataset, run, initial, selectedSource, stores }: {
  dataset: string;
  run: string;
  initial: ActionsInfo;
  selectedSource: string | null;
  stores: string[];              // runtime stores the page knows (for result links)
}) {
  const { info, setInfo, refresh } = useActions(dataset, run, initial);

  function upsert(job: Job) {
    setInfo((cur) => ({
      ...cur,
      busy: live(job) ? { id: job.id, run: job.run, label: job.label } : cur.busy?.id === job.id ? null : cur.busy,
      jobs: [job, ...cur.jobs.filter((j) => j.id !== job.id)],
    }));
    if (!live(job)) refresh();
  }

  return (
    <>
      <h3>Runtime sweep</h3>
      <p className="muted small">
        Executes every pipeline of a source under stratum (<span className="mono">pipeline_analyzer.runtime</span>)
        and records wall time, per-operator time and memory in a runtime store beside the run. The store is a
        cache: a pipeline already measured with the same code, data size and stratum build is skipped, and
        progress is saved after each pipeline, so a stopped sweep resumes where it left off.
      </p>
      <SweepForm dataset={dataset} run={run} info={info} selectedSource={selectedSource} onStarted={upsert} />
      {info.jobs.length > 0 && (
        <>
          <h3>Recent jobs</h3>
          {info.jobs.map((j) => <JobCard key={j.id} job={j} stores={stores} onChange={upsert} />)}
        </>
      )}
    </>
  );
}
