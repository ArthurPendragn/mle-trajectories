import { shortCommit } from "@/lib/format";
import type { RuntimeStore } from "@/lib/types";

/** A quiet note that a store was measured under another stratum build or
 *  older code. Informational only: the numbers stay visible. */
export function StaleHint({ store, current, compact = false }: {
  store: RuntimeStore;
  current: string | null;
  compact?: boolean;
}) {
  const oldBuild = store.n_old_build > 0;
  const changed = store.n_code_changed > 0;
  if (!oldBuild && !changed) return null;
  const known = store.commits.filter((c) => c !== "unknown").map(shortCommit);
  const built = known.length > 0 ? `stratum ${known.join(", ")}`
    : "a stratum build that was not recorded";
  const parts = [];
  if (oldBuild) {
    parts.push(`${store.n_old_build} measured under ${built}, installed is ${shortCommit(current)}`);
  }
  if (changed) parts.push(`${store.n_code_changed} pipeline file(s) changed since measuring`);
  const text = `${parts.join("; ")}. Re-measure for numbers comparable with the current build.`;
  if (compact) {
    return <span className="hint" title={text}>ⓘ older build</span>;
  }
  return <p className="hint">ⓘ {text}</p>;
}
