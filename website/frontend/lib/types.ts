// Shapes returned by website/backend/app.py.

export type Metric = { name: string | null; lower_is_better: boolean | null };

export type Source = {
  name: string;
  label: string | null;
  skrub: boolean;
  dirs: string[];
  default: boolean;
  hidden: boolean;
  fold_identical_code: boolean;
  note: string | null;
  n_files: number;
  coverage: [number, number];
};

export type RuntimeStore = {
  file: string;
  source: string | null;
  data: string;                 // "input" or the sample folder it ran on
  legacy_rows: number | null;   // measured with the removed --sample-rows cap
  n_ok: number;
  n_failed: number;
  n_code_changed: number;
  n_old_build: number;
  n_data_changed: number;       // the sample was rebuilt differently since
  commits: string[];
};

export type RunSummary = {
  id: string;
  dataset: string;
  name: string;
  label: string;
  agent: string | null;
  metric: Metric;
  lineage: "trajectory" | "results.json" | null;
  n_steps: number;
  best: { module: string | null; score: number } | null;
  default_source: Source | null;
  n_sources: number;
  runtime: RuntimeStore[];
  n_warnings: number;
};

export type DatasetInfo = {
  name: string;
  label: string;
  task: string | null;
  data: string;
  data_status: "local" | "remote" | "missing";
  samples: string[];
  has_sample_recipe: boolean;
  note: string | null;
  metric: Metric;
  warnings: string[];
};

export type Corpus = {
  stratum_commit: string | null;
  datasets: (DatasetInfo & { runs: RunSummary[] })[];
};

export type Step = {
  module: string | null;
  parent: string | null;
  phase: string | null;
  score: number | null;
  desc: string | null;
};

export type RunDetail = Omit<RunSummary, "runtime"> & {
  runtime: RuntimeStoreDetail[];
  selection: Selection;
  note: string | null;
  path: string;
  dataset_info: DatasetInfo;
  stratum_commit: string | null;
  trajectory: { file: string; meta: Record<string, string> } | null;
  originals: Source | null;
  sources: Source[];
  steps: Step[];
  warnings: string[];
};

export type Selection = {
  source: { name: string; reason: string } | null;
  runtime: { name: string; reason: string } | null;
};

export type RuntimeStoreDetail = RuntimeStore & {
  name: string;
  label: string | null;
  note: string | null;
  hidden: boolean;
  measured_at: string | null;
};

export type TreeData = {
  modules: string[];
  phase_colors: Record<string, string>;
  svg: { delta?: string; phase?: string };
  lower_is_better?: boolean;
};

export type RuntimeProfile = {
  file: string;
  source: string | null;
  data: string;
  legacy_rows: number | null;
  n_measured: number;
  n_failed: number;
  wall_total_s: number;
  op_total_s: number;
  scoring: string | null;
  pipelines: {
    name: string; status: string; wall_s: number | null; op_time_s: number | null;
    total_s: number | null; max_rss_mb: number | null; mean_mb: number | null;
    n_op_calls: number | null; best_score: number | null; error: string[] | null;
    code_changed: boolean; old_build: boolean;
  }[];
  ops: {
    op: string; time_s: number; share: number | null; calls: number;
    per_call_s: number | null; n_pipelines: number;
    heaviest: { name: string; time_s: number };
  }[];
};

/** pipeline_analyzer.merged payload (short keys: it was designed to be inlined). */
export type MergedPayload = {
  pipelines: {
    n: string; ph: string; c: string; s: number | null; d: number | null; up: number | null;
    p: number | null; ok: boolean; ops: number; root: number | null; desc: string | null;
  }[];
  nodes: { l: string; t: string; f: string; e: string | null; i: number[]; m: number[] }[];
  phaseColors: Record<string, string>;
  lowerIsBetter: boolean;
};

export type OpStats = {
  n_pipelines: number;
  rows: { op: string; total: number; present: number; mean: number; median: number;
          std: number; min: number; max: number }[];
  sizes: { total: number; median: number; min: number; max: number } | null;
};

export type AnalysisData = {
  run: string;
  source: string;
  generated_at: string;
  stratum_commit: string | null;
  elapsed_s: number;
  n_pipelines: number;
  merged: MergedPayload;
  stats: { logical: OpStats; physical: OpStats; physical_missing: number };
  failed: { name: string; error: string }[];
  folded: [string, string][];
};

export type AnalysisStatus =
  | { status: "ready"; data: AnalysisData }
  | { status: "running" | "queued" | "missing"; log?: string }
  | { status: "failed"; log: string };

// --- actions -------------------------------------------------------------- //
export type SweepParams = {
  source: string;
  data: string;
  timeout_s: number;
  retry_failed: boolean;
  force: boolean;
};

export type DataOption = { name: string; label: string; ok: boolean; note: string | null };

export type SweepPlan = {
  params: SweepParams;
  store: string;
  store_exists: boolean;
  store_summary: { n_ok: number; n_failed: number; n_stale: number } | null;
  n_pipelines: number;
  command: string;
  listing?: { ok: boolean; text: string; would_run: number | null; total: number | null };
};

export type SampleParams = { size: string; force: boolean };

export type SampleInfo = {
  name: string;
  manifest: boolean;
  size?: number | null;
  built?: string | null;
  adopted?: boolean;
  recipe_sha1?: string | null;
  note?: string | null;
  rows?: Record<string, [number | null, number | null]>;   // file -> [out, src]
};

export type SamplePlan = {
  dataset: string;
  name: string;
  size: number;
  exists: boolean;
  recipe_sha1: string;
  target: string | null;
  script: string | null;
  tables: Record<string, string>;
  force: boolean;
  command: string;
};

export type JobState = "starting" | "running" | "done" | "failed" | "stopped" | "lost";

export type Job = {
  id: string;
  action: "runtime-sweep" | "build-sample" | string;
  run: string;
  dataset: string | null;
  label: string;
  params: Partial<SweepParams> & { name?: string; size?: number; force?: boolean };
  command: string;
  outputs: { runtime?: string; sample?: string };
  started_at: string;
  finished_at: string | null;
  returncode: number | null;
  state: JobState;
  progress: {
    total: number | null; todo: number | null; cached: number | null;
    done: number; failed: number; current: string | null; finished: boolean;
  };
  log: string;
};

export type ActionsInfo = {
  sweep: {
    sources: { name: string; label: string | null; coverage: [number, number]; default: boolean }[];
    data: DataOption[];
  };
  sample: {
    ok: boolean;
    reason: string | null;
    dataset: string;
    target: string | null;
    script: string | null;
    samples: SampleInfo[];
  };
  jobs: Job[];
  busy: { id: string; run: string; label: string } | null;   // a sweep running anywhere
};

export type JobList = { jobs: Job[]; n_live: number };
