"use client";

import { useEffect, useState } from "react";
import { planSweepAction, startSweepAction } from "@/app/actions/run-actions";
import type { ActionsInfo, Job, SweepParams, SweepPlan } from "@/lib/types";

/** Run pipeline_analyzer.runtime over one source, on the full data or a sample. */
export function SweepForm({ dataset, run, info, selectedSource, onStarted }: {
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
  const [timeoutMin, setTimeoutMin] = useState("60");
  const [retryFailed, setRetryFailed] = useState(false);
  const [force, setForce] = useState(false);

  const [plan, setPlan] = useState<SweepPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [listing, setListing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [starting, setStarting] = useState(false);

  // a sample built meanwhile shows up in the options; keep the choice valid
  useEffect(() => {
    if (!data.some((d) => d.name === dataName && d.ok)) {
      setDataName(data.find((d) => d.ok)?.name ?? "input");
    }
  }, [data, dataName]);

  const params: SweepParams = {
    source, data: dataName,
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
                {d.label}{d.ok ? "" : ` — ${d.note ?? "unavailable"}`}
              </option>
            ))}
          </select>
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
      {data.length === 1 && (
        <p className="muted small">No sample of this dataset yet: build one above for a quicker sweep.</p>
      )}

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
              Measure {count ?? `up to ${plan?.n_pipelines}`} pipeline(s) on {dataOpt?.label},
              each for at most {timeoutMin} min?
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
