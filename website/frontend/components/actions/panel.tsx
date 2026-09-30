"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import type { ActionsInfo, Job } from "@/lib/types";
import { JobCard, live } from "./job-card";
import { SampleForm } from "./sample-form";
import { SweepForm } from "./sweep-form";

const POLL_MS = 3000;

/** The run page's actions: build a sample of the dataset, run a runtime sweep,
 *  and the jobs they started. Polls while a job runs and re-renders the page
 *  when one ends, so a new sample or runtime store shows up. */
export function ActionsPanel({ dataset, run, initial, selectedSource, stores, sweepReason }: {
  dataset: string;
  run: string;
  initial: ActionsInfo;
  selectedSource: string | null;
  stores: string[];
  sweepReason: string | null;      // why a sweep is unavailable, or null
}) {
  const router = useRouter();
  const [info, setInfo] = useState(initial);
  useEffect(() => setInfo(initial), [initial]);
  const wasLive = useRef(initial.jobs.some(live));
  const anyLive = info.jobs.some(live);

  useEffect(() => {
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

  function upsert(job: Job) {
    setInfo((cur) => ({
      ...cur,
      busy: job.action !== "runtime-sweep" ? cur.busy
        : live(job) ? { id: job.id, run: job.run, label: job.label }
        : cur.busy?.id === job.id ? null : cur.busy,
      jobs: [job, ...cur.jobs.filter((j) => j.id !== job.id)],
    }));
    if (!live(job)) router.refresh();
  }

  return (
    <>
      <h3>Build sample</h3>
      <p className="muted small">
        A smaller copy of <span className="mono">{dataset}/input/</span> as its own folder{" "}
        (<span className="mono">sample_&lt;size&gt;/input/</span>, same file names), built from the{" "}
        <span className="mono">[sample]</span> recipe in <span className="mono">dataset.toml</span> by{" "}
        <span className="mono">tools/dataset_sample</span>. Pipelines run on it unmodified; nothing is
        patched.
      </p>
      {info.sample.ok
        ? <SampleForm dataset={dataset} run={run} info={info.sample} onStarted={upsert} />
        : <p className="muted">Unavailable: {info.sample.reason}.</p>}

      <h3>Runtime sweep</h3>
      <p className="muted small">
        Executes every pipeline of a source under stratum (<span className="mono">pipeline_analyzer.runtime</span>)
        and records wall time, per-operator time and memory in a runtime store beside the run. The store is a
        cache: a pipeline already measured with the same code, data and stratum build is skipped, and
        progress is saved after each pipeline, so a stopped sweep resumes where it left off.
      </p>
      {sweepReason
        ? <p className="muted">Unavailable: {sweepReason}.</p>
        : <SweepForm dataset={dataset} run={run} info={info} selectedSource={selectedSource}
                     onStarted={upsert} />}

      {info.jobs.length > 0 && (
        <>
          <h3>Recent jobs</h3>
          {info.jobs.map((j) => <JobCard key={j.id} job={j} stores={stores} onChange={upsert} />)}
        </>
      )}
    </>
  );
}
