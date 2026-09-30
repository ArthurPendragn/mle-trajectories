"use client";

import { useEffect, useState } from "react";
import { planSampleAction, startSampleAction } from "@/app/actions/run-actions";
import type { ActionsInfo, Job, SamplePlan } from "@/lib/types";

const PRESETS = ["10k", "100k", "1m"];

/** Build a persisted sample of the run's dataset (tools/dataset_sample). */
export function SampleForm({ dataset, run, info, onStarted }: {
  dataset: string;
  run: string;
  info: ActionsInfo["sample"];
  onStarted: (job: Job) => void;
}) {
  const [size, setSize] = useState("100k");
  const [force, setForce] = useState(false);
  const [plan, setPlan] = useState<SamplePlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    let stale = false;
    setConfirming(false);
    const t = setTimeout(async () => {
      const res = await planSampleAction(dataset, run, { size, force });
      if (stale) return;
      setPlan(res.ok ? res.value : null);
      setError(res.ok ? null : res.error);
    }, 250);
    return () => {
      stale = true;
      clearTimeout(t);
    };
  }, [dataset, run, size, force]);

  async function start() {
    setStarting(true);
    try {
      const res = await startSampleAction(dataset, run, { size, force });
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

  const blocked = !plan || !!error || (plan.exists && !force);

  return (
    <div className="sweep">
      <div className="sweep-form">
        <label>
          <span className="muted">rows in the sampled tables</span>
          <span className="sweep-cap">
            <input value={size} onChange={(e) => setSize(e.target.value)} placeholder="100k" />
            {PRESETS.map((p) => (
              <button key={p} className={p === size ? "on" : ""} onClick={() => setSize(p)}>{p}</button>
            ))}
          </span>
        </label>
        {plan?.exists && (
          <label className="check" title="build it again and replace the existing folder once the new one passed its checks">
            <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
            rebuild (replaces {plan.name}/)
          </label>
        )}
      </div>
      {error && <p className="error small">{error}</p>}
      {plan && !error && (
        <div className="small">
          <p>
            Writes <span className="mono">{dataset}/{plan.name}/</span>
            {plan.exists && !force && <span className="partial"> — exists already</span>}.{" "}
            {plan.script
              ? <>Built by the dataset&apos;s own <span className="mono">{plan.script}</span>.</>
              : <>
                  {Object.entries(plan.tables).map(([f, rule], i) => (
                    <span key={f}>{i > 0 && " · "}<span className="mono">{f}</span> <span className="muted">{rule}</span></span>
                  ))}
                  {plan.target && <> · stratified on <span className="mono">{plan.target}</span></>}
                </>}
          </p>
          <p className="mono muted">{plan.command}</p>
        </div>
      )}
      <div className="sweep-buttons">
        {!confirming ? (
          <button className="primary" onClick={() => setConfirming(true)} disabled={blocked}>
            build sample…
          </button>
        ) : (
          <>
            <span className="small">
              Build {plan?.name} ({plan?.size.toLocaleString()} rows){force ? ", replacing the existing one" : ""}?
            </span>
            <button className="primary" onClick={start} disabled={starting}>
              {starting ? "starting…" : "build"}
            </button>
            <button onClick={() => setConfirming(false)}>cancel</button>
          </>
        )}
      </div>
    </div>
  );
}
