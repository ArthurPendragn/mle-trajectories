import type { RunDetail } from "./types";

// What a run supports, derived from what exists for it. Each analysis or
// action states its requirement once here; the UI shows unavailable ones
// greyed out with the reason instead of letting them fail halfway.

export type Capability = {
  key: string;
  label: string;
  kind: "analysis" | "action";
  ok: boolean;
  reason?: string;        // why it is unavailable
  planned?: boolean;      // not implemented yet
};

function hasPlans(run: RunDetail): boolean {
  return run.sources.some((s) => !s.hidden && s.coverage[0] > 0);
}

export function capabilities(run: RunDetail): Capability[] {
  const plans = hasPlans(run);
  const data = run.dataset_info.data_status !== "missing" || run.dataset_info.samples.length > 0;
  const ds = run.dataset_info;
  return [
    {
      key: "sample", label: "Build sample", kind: "action",
      ok: ds.data_status === "local" && ds.has_sample_recipe,
      reason: !ds.has_sample_recipe ? "no [sample] recipe in dataset.toml"
        : ds.data_status !== "local" ? "needs the dataset's input/ on this machine" : undefined,
    },
    {
      key: "sweep", label: "Runtime sweep", kind: "action", ok: plans && data,
      reason: !plans ? "needs skrub DataOps plans"
        : !data ? "needs the dataset's input/ (or a sample) on this machine" : undefined,
    },
  ];
}

/** Why the operator analyses cannot be shown, or null when they can. */
export function noPlansReason(run: RunDetail): string | null {
  if (hasPlans(run)) return null;
  return run.sources.length === 0
    ? "No skrub DataOps plans yet: skrubify the agent's original scripts first."
    : "The pipeline sources hold no plan for any pipeline of this run.";
}
