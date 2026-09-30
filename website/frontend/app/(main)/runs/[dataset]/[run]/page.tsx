import Link from "next/link";
import { Suspense } from "react";
import { notFound } from "next/navigation";
import { getActions, getAnalysis, getRun, getRuntimeProfile, getTree } from "@/lib/dal";
import { capabilities, noPlansReason } from "@/lib/capabilities";
import { firstLine, fmtScore, improvement, metricLabel } from "@/lib/format";
import type { RunDetail, Source } from "@/lib/types";
import { RunAnalysis } from "@/components/run-analysis";
import { RuntimeProfile } from "@/components/runtime-profile";
import { RuntimeSweep } from "@/components/runtime-sweep";
import { SelectionBar } from "@/components/selection-bar";
import { StaleHint } from "@/components/stale-hint";

export const dynamic = "force-dynamic";

function one(v: string | string[] | undefined): string | undefined {
  return Array.isArray(v) ? v[0] : v;
}

function SourceRow({ s, run }: { s: Source; run: RunDetail }) {
  const [have, total] = s.coverage;
  const selected = s.name === run.selection.source?.name;
  return (
    <tr className={s.hidden ? "muted" : ""}>
      <td>
        {s.name}
        {selected && <span className="badge">selected</span>}
        {s.label && <div className="muted small">{s.label}</div>}
      </td>
      <td className="num"><span className={have === total ? "" : "partial"}>{have}/{total}</span></td>
      <td className="small">{s.dirs.join(", ")}</td>
      <td className="small">
        {s.fold_identical_code && <div>fold identical code</div>}
        {s.note}
        {s === run.originals && !s.skrub && <span className="muted">agent&apos;s original scripts</span>}
      </td>
    </tr>
  );
}

function Steps({ run }: { run: RunDetail }) {
  const score = new Map(run.steps.filter((s) => s.module).map((s) => [s.module!, s.score]));
  const best = run.best?.module;
  return (
    <table>
      <thead>
        <tr>
          <th>#</th><th>Pipeline</th><th>Phase</th><th>Parent</th>
          <th className="num">Score</th><th className="num">Δ parent</th><th>Description</th>
        </tr>
      </thead>
      <tbody>
        {run.steps.map((s, i) => {
          const d = improvement(s.score, s.parent ? score.get(s.parent) ?? null : null, run.metric);
          return (
            <tr key={i} id={s.module ? `step-${s.module}` : undefined}
                className={s.module === best ? "best" : s.module ? "" : "muted"}>
              <td className="num muted">{i + 1}</td>
              <td className="mono small">{s.module ?? "(no code kept)"}</td>
              <td>{s.phase ?? ""}</td>
              <td className="mono small muted">{s.parent ?? ""}</td>
              <td className="num">{fmtScore(s.score)}</td>
              <td className={`num ${d === null ? "" : d > 0 ? "up" : d < 0 ? "down" : "muted"}`}>
                {d === null ? "" : `${d > 0 ? "+" : ""}${fmtScore(d)}`}
              </td>
              <td className="small" title={s.desc ?? undefined}>{firstLine(s.desc)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default async function RunPage({ params, searchParams }: PageProps<"/runs/[dataset]/[run]">) {
  const { dataset, run: name } = await params;
  const q = await searchParams;
  const run = await getRun(dataset, name, { source: one(q.source), runtime: one(q.runtime) });
  if (!run) notFound();
  const ds = run.dataset_info;
  const src = run.selection.source?.name ?? null;
  const rtName = run.selection.runtime?.name ?? null;
  const noPlans = noPlansReason(run);

  // the operator analysis is started here (idempotent) so it is building by the
  // time the page is on screen; the tree and runtime profile need no build
  const [tree, analysis, profile, actionsInfo] = await Promise.all([
    getTree(dataset, name, src ?? undefined),
    src && !noPlans ? getAnalysis(dataset, name, src, { start: true }) : Promise.resolve(null),
    rtName ? getRuntimeProfile(dataset, name, rtName) : Promise.resolve(null),
    getActions(dataset, name),
  ]);
  const store = run.runtime.find((r) => r.name === rtName) ?? null;
  const actions = capabilities(run);
  const sweep = actions.find((a) => a.key === "sweep");

  return (
    <>
      <p className="crumbs"><Link href="/">Corpus</Link> / {ds.label}</p>
      <h1>
        {run.label} <span className={`badge agent-${run.agent}`}>{run.agent ?? "agent ?"}</span>
      </h1>
      <p className="muted mono small">{run.path}</p>
      {run.note && <p className="note">{run.note}</p>}

      <div className="facts">
        <div><span className="muted">metric</span>{metricLabel(run.metric)}</div>
        <div><span className="muted">lineage</span>
          {run.lineage ? `${run.lineage}${run.trajectory ? ` (${run.trajectory.file})` : ""}` : "none"}</div>
        <div><span className="muted">steps</span>{run.n_steps}</div>
        <div><span className="muted">best</span>
          {fmtScore(run.best?.score)} <span className="mono small muted">{run.best?.module}</span></div>
        <div><span className="muted">data</span>
          {ds.data_status}{ds.samples.length > 0 && ` · samples: ${ds.samples.join(", ")}`}</div>
        <div><span className="muted">actions</span>
          <span>
            {actions.map((a) => a.ok
              ? <a key={a.key} className="action" href={`#action-${a.key}`}>{a.label}</a>
              : <button key={a.key} className="action" disabled title={a.reason}>{a.label}</button>)}
          </span>
        </div>
      </div>

      <Suspense>
        <SelectionBar sources={run.sources} runtime={run.runtime} selection={run.selection} />
      </Suspense>

      <RunAnalysis dataset={dataset} run={name} source={noPlans ? null : src}
                   noSourceReason={noPlans ?? ""} tree={tree} initial={analysis} />

      <section id="action-sweep">
        <h2>Actions</h2>
        {!sweep?.ok || !actionsInfo
          ? <p className="muted">Runtime sweep unavailable: {sweep?.reason ?? "unknown run"}.</p>
          : <RuntimeSweep dataset={dataset} run={name} initial={actionsInfo} selectedSource={src}
                          stores={run.runtime.map((r) => r.name)} />}
      </section>

      <section id="runtime-profile">
        <h2>Runtime profile</h2>
        {!profile || !store
          ? <p className="muted">No runtime measurements for this run yet.</p>
          : <RuntimeProfile profile={profile} store={store} current={run.stratum_commit} />}
      </section>

      <section>
        <h2>Pipeline sources</h2>
        <table>
          <thead><tr><th>Source</th><th className="num">Coverage</th><th>Folders</th><th /></tr></thead>
          <tbody>
            {run.originals && !run.originals.skrub && <SourceRow s={run.originals} run={run} />}
            {run.sources.map((s) => <SourceRow key={s.name} s={s} run={run} />)}
          </tbody>
        </table>
      </section>

      <section>
        <h2>Runtime stores</h2>
        {run.runtime.length === 0 ? <p className="muted">None yet.</p> : (
          <table>
            <thead>
              <tr><th>Store</th><th>Source</th><th className="num">OK</th>
                <th className="num">Failed</th><th>Data</th><th /></tr>
            </thead>
            <tbody>
              {run.runtime.map((r) => (
                <tr key={r.file} className={r.hidden ? "muted" : ""}>
                  <td className="mono small">
                    {r.file}{r.name === rtName && <span className="badge">selected</span>}
                  </td>
                  <td>{r.source ?? <span className="muted">?</span>}</td>
                  <td className="num">{r.n_ok}</td>
                  <td className="num">{r.n_failed || ""}</td>
                  <td>
                    {r.data === "input" ? (r.sample_rows ? "input/" : "full data") : `${r.data}/`}
                    {r.sample_rows ? `, ${r.sample_rows.toLocaleString()} rows` : ""}
                  </td>
                  <td><StaleHint store={r} current={run.stratum_commit} compact /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {run.trajectory && (
        <section>
          <h2>Trajectory</h2>
          <dl className="meta">
            {Object.entries(run.trajectory.meta).map(([k, v]) => (
              <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
            ))}
          </dl>
        </section>
      )}

      <section>
        <h2>Steps</h2>
        {run.steps.length === 0 ? <p className="muted">No lineage recorded for this run.</p>
          : <Steps run={run} />}
      </section>

      {(run.warnings.length > 0 || ds.warnings.length > 0) && (
        <section>
          <h2>Warnings</h2>
          <ul>{[...ds.warnings, ...run.warnings].map((w) => <li key={w}>{w}</li>)}</ul>
        </section>
      )}
    </>
  );
}
