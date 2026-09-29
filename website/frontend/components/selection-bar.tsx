"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type { RuntimeStoreDetail, Selection, Source } from "@/lib/types";

const REASONS: Record<string, string> = {
  "selected": "your choice",
  "only one": "the only one",
  "run.toml": "default from run.toml",
  "dataset.toml": "default from dataset.toml",
};

function Reason({ why }: { why: string }) {
  return <span className="muted small">{REASONS[why] ?? why}</span>;
}

/** Which pipeline source and runtime store the analyses below use. The choice
 *  lives in the URL (?source=&runtime=), so a view can be linked. */
export function SelectionBar({ sources, runtime, selection }: {
  sources: Source[];
  runtime: RuntimeStoreDetail[];
  selection: Selection;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();

  function choose(key: "source" | "runtime", value: string) {
    const q = new URLSearchParams(params.toString());
    q.set(key, value);
    router.push(`${pathname}?${q.toString()}`, { scroll: false });
  }

  const visibleSources = sources.filter((s) => !s.hidden || s.name === selection.source?.name);
  const visibleStores = runtime.filter((r) => !r.hidden || r.name === selection.runtime?.name);

  return (
    <div className="selection">
      <label>
        <span className="muted">skrub plans</span>
        {selection.source ? (
          <>
            <select value={selection.source.name} onChange={(e) => choose("source", e.target.value)}
                    disabled={visibleSources.length < 2}>
              {visibleSources.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name === "pipelines" && s.skrub ? "agent's plans" : s.label ?? s.name}
                  {" "}({s.coverage[0]}/{s.coverage[1]})
                </option>
              ))}
            </select>
            <Reason why={selection.source.reason} />
          </>
        ) : <span className="muted">none yet</span>}
      </label>
      <label>
        <span className="muted">runtime store</span>
        {selection.runtime ? (
          <>
            <select value={selection.runtime.name} onChange={(e) => choose("runtime", e.target.value)}
                    disabled={visibleStores.length < 2}>
              {visibleStores.map((r) => (
                <option key={r.name} value={r.name}>
                  {r.label ?? r.name.replace(/^runtime_stats_/, "")}
                  {" · "}{r.sample_rows ? `${r.sample_rows.toLocaleString()} rows` : "full data"}
                </option>
              ))}
            </select>
            <Reason why={selection.runtime.reason} />
          </>
        ) : <span className="muted">none yet</span>}
      </label>
    </div>
  );
}
