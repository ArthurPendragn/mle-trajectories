# Finding optimization targets for stratum

## Objective and decision

This corpus contains ML engineering agents' search trajectories: sequences of
pipelines for different datasets and tasks. Their skrub DataOps representations
expose computation graphs that we can execute and profile.

The objective is to identify operations and recurring subgraphs for which a
stratum improvement would save substantial time across representative searches.
The deliverable is an evidence-backed optimization backlog: what to improve,
which inputs and kwargs to support, how broadly it applies, and the expected
end-to-end benefit.

**Use stratum for primary bottleneck discovery, with skrub as the reference
baseline.** If resources initially allow only one engine, profile stratum.

| Execution | Question answered |
| --- | --- |
| Plain skrub | Where does the reference execution spend time? |
| Current stratum, with the intended runtime configuration | What remains expensive for stratum users? |
| Paired executions of the same workload | What does stratum improve, miss, or make slower? |
| Stratum with one optimization changed | Which mechanism explains a difference? |

Skrub timings alone can prioritize work that stratum already eliminates or
accelerates. Stratum timings alone cannot establish speedup over skrub. Record
both when making comparative claims, but do not require an exhaustive skrub
sweep before investigating an expensive stratum component.

## Define the workloads

Measure these workloads separately:

1. **Independent pipeline evaluation.** Start a new process for each pipeline
   and execute its declared evaluation. This discovers costly implementations,
   data preparation, and engine overhead. A new process resets process-local
   state; it does not guarantee a cold filesystem or remote-object cache.
2. **Agent search replay.** Execute the actual ordered or batched evaluations
   from a trajectory, preserving the runtime state that the intended stratum
   integration would preserve. This measures reuse and repeated-work avoidance
   across candidates. Declare the process, session, and cache lifetime.

Search replay must respect which candidates were available together. Giving the
engine all future pipelines in advance measures an offline batch opportunity;
label that separately from an online agent search.

Distinguish several forms of reuse: shared branches inside a DAG, candidates
inside one grid search, and separate pipeline evaluations across a trajectory.
An expensive operation executed independently in many pipelines is evidence of
a possible reuse opportunity, not proof that the work is semantically reusable.
Training rows, fold identities, targets, parameters, random state, and upstream
data must agree wherever reuse requires them to agree.

## What the existing website already provides

The website is a useful starting point for discovery. Reuse its runtime sweep
and result browsing rather than building a second job system.

This assessment is based on repository source and two existing runtime stores,
reviewed on 2026-11-11. No new pipeline benchmark was executed for this document.
The installed stratum package and `pyproject.toml` both identify commit
`4c56c54b35bf43b19be2e5128230e2d9f1ef521c`. Recheck behavior when changing the pin.

| Existing capability | Value for this work |
| --- | --- |
| Run page → Actions → Runtime sweep | Executes a selected DataOps source against full data or a sample; supports timeout, retry, resume, and remeasurement. |
| Detached jobs, progress, logs, cancellation | Supports long sweeps without tying their lifetime to a browser session. The website permits one runtime sweep at a time. |
| One child process per pipeline | Isolates failures and process-local state; provides independent evaluation, rather than persistent search replay. |
| Stratum scheduler with `stats=True` | Measures physical operator bodies and aggregates their time and call counts by readable label. |
| Runtime profile | Shows per-pipeline wall time, operator time, memory, and CV score; ranks operator classes by total time, share, calls, per-call time, and pipeline coverage. |
| Runtime JSON stores | Retain more detail than the class table, including operator labels, buffer overhead, pool counters, scores, and measurement metadata. |
| Static operator explorer | Helps inspect shared structure and compare pipelines. Structural sharing in this display does not establish execution-time reuse. |
| Logical and best-effort physical DAG extraction | Provides structural context; a separately extracted physical graph is not necessarily the exact plan used by a measured execution. |
| Sample fingerprints and per-entry stratum commits | Catch some stale measurements and support resumable sweeps. |

The execution path is:

```text
website runtime-sweep action
    → pipeline_analyzer.runtime
        → new pipeline_analyzer._measure child for each pipeline
            → stratum._api.grid_search
                → optimizer and scheduler
```

The child intercepts `make_grid_search` and runs with eager DataOps previews
disabled. Consequently, importing a pipeline that says `import skrub` does not
make this a skrub baseline. The existing sweep always routes scoring through
stratum.

The measurements have different boundaries:

- `wall_s`: the intercepted stratum grid-search call, including its optimization
  and execution, and enabled statistics reporting.
- `op_time_s`: the sum of recorded physical operator-body durations.
- `buffer_overhead_s`: scheduler input/output buffer work recorded separately.
- `total_s`: pipeline-file execution and scoring inside the measurement child;
  includes additional work the script performs within that window.
- `elapsed_s`: the parent's observation of the child lifetime, including startup.

The inspected stratum build uses a sequential scheduler, although libraries
called by an operator can use multiple threads or processes. Operator time is
elapsed time around `op.process`, not CPU core-seconds. Do not interpret operator
share as the fraction of total wall time without checking the timing boundaries.

### Important gaps before using this as a benchmark suite

**Attribution is too coarse for deciding which kernel to implement.** Generic
labels such as `Predictor [df]` combine different estimators and phases, and
identical label strings can combine distinct nodes. Some implementations already
have useful labels, such as `LibXGBoost`; preserve that distinction. The website
then aggregates further by operator class. It does not currently join each
measurement to estimator kwargs, actual input shape, or a particular subgraph.

**Stratum already provides a richer diagnostics path.** At the inspected pin,
`stratum.dev_config(diagnostics_dir=...)` enables versioned JSON exports with
optimizer plan snapshots, plan-local node IDs and implementations, per-node
processing and buffer times broken down by mode and fold, and optimization and
execution timings. The runtime runner now exposes these through
`--diagnostics-dir`, and the [HPC adapter](hpc/README.md) enables them for profile
cells. Continue by enriching the exports with missing estimator and input
metadata. Plan-local IDs need an explicit mapping or signature to compare nodes
across plans; snapshots alone do not establish logical-to-physical provenance.

**The cache key is not a complete experiment identity.** `is_fresh` checks the
pipeline file hash, sample fingerprint, stratum commit, and whether memory
samples exist. It does not include shared helper code, scorer, statistics mode,
all runtime flags, hardware, thread settings, or dependency versions. Full data
currently has no content fingerprint in this check. Metadata at store level can
also be rewritten while older entries remain. Use separate stores and an
explicit experiment manifest until this is strengthened.

**Stale results remain in displayed totals.** The website flags changed code or
old builds, but its runtime profile aggregates every entry with `status="ok"`.
It does not exclude stale entries from the operator ranking. Filter to a single
validated measurement cohort before comparing or aggregating results.

**Scoring documentation and flags need updating.** The analyzer README describes
an older scheduler's scoring failures. The installed stratum source now resolves
declared scoring, scorer kwargs, and negative-score signs; it rejects missing
scoring unless the plan declares it. The runner's `scoring_honoured` helper still
returns true only for strings and does not prove correctness. Check the installed
behavior and equivalent predictions/scores, rather than treating that flag or
the old README as authoritative. For plan-only files, ensure a supplied default
does not override `.skb.with_scoring`.

**The runner intercepts only part of the scoring API.** It forwards the DAG,
explicit CV, and scorer to stratum's grid search, rather than reproducing every
original search option. It retains the last successful scoring call if a script
scores multiple plans. Inventory these cases and represent each evaluation
explicitly; do not mistake one stored entry for an arbitrary script's full cost.

**Repetitions, a plain skrub baseline, and persistent trajectory replay are
missing.** Re-measurement replaces an entry rather than retaining a distribution.
The website does not expose engine choice, thread budgets, or stratum runtime
configuration as experiment dimensions. These require extensions, not different
interpretations of the existing stores.

## Measurement protocol

### 1. Select representative, valid pipelines

Start with a stratified pilot across datasets, agents, model families, input
sizes, and common transformation patterns. Include early and later candidates,
branches, ensembles, and repeated variants; selecting only final winners would
miss much of an agent search's execution cost.

Use one canonical DataOps conversion per original pipeline. Multiple skrubify
outputs are alternative representations, not additional workload occurrences.
For independent discovery, deduplicate equivalent pipelines while retaining
their occurrence counts. For trajectory replay, retain actual repeated attempts.
Report both trajectory-weighted cost and breadth across distinct datasets/runs,
so one long trajectory does not determine every priority.

Before timing, establish that the conversion preserves the intended algorithm,
data, preprocessing, parameters, fold identities, and early-stopping behavior.
Record any intentional leakage correction; compare engines on the same corrected
plan. Leave agent originals under `pipelines/` untouched.

Use persisted, self-consistent samples for iteration and validate candidates on
representative larger or full data. Do not extrapolate a tiny-sample ranking
linearly: cardinality, join fan-out, sparse/dense output, memory pressure, and
model training costs can change the ranking.

### 2. Lock the experiment and evaluation semantics

Give every experiment a manifest containing:

| Dimension | Record |
| --- | --- |
| Workload | Dataset/run/pipeline IDs, canonical source, occurrence weights, candidate order or batches, workload mode. |
| Code | Repository revision, pipeline and transitive helper hashes, measurement-tool revision. |
| Data | Content/version identity, sample recipe and seed, actual files or objects read, row counts. |
| Engine | Skrub version or stratum commit/build identity, Python and library versions, effective optimizer/runtime flags and selected backends. |
| Resources | Machine/CPU, available cores and memory, GPU if used, thread/process limits, memory-pool budget. |
| Evaluation | Actual train/test indices or their hashes, fold count, grid/choices, seeds, scorer and kwargs, response method, early stopping, refit policy. |
| Measurement | Repetition, ordering, cache policy, stats/diagnostics/memory settings, timing boundaries. |

Preserve CV declared on `mark_as_X(cv=..., split_kwargs=...)`; do not silently
replace it with a harness-level `cv=`. Preserve raw-target scoring and any target
inversion. A first-fold holdout must remain one fold, rather than becoming full
k-fold CV. Keep `refit=False` where the workload only evaluates candidates.

For the plain skrub baseline, use an isolated process that does not import
stratum or install its patches. A pipeline that imports stratum needs an
explicit harness-level import adaptation or benchmark copy, preserving its
semantics and recording the adaptation. Merely setting `scheduler=False` in an
already patched process is insufficient evidence of an unmodified baseline.

Verify that changing `--run-in` actually changes the inputs. It changes the
working directory; it cannot redirect hard-coded absolute paths or `gs://`
objects. Remote-read pipelines may still read full data when a sample folder is
selected. Record this explicitly, and separate local-compute experiments from
remote-I/O experiments when interpreting results.

### 3. Collect detailed profiles and clean elapsed-time measurements

Use detailed stratum profiles to locate candidates, and separate executions
with profiling disabled to validate speedup. Keep the intended optimizer and
implementation selection enabled for the primary profile. Disabling particular
features is a follow-up experiment to explain a result.

For each executed node, retain:

- Execution/run ID, physical node ID, operator/implementation class, upstream
  edges, and available correspondence to logical operations.
- Estimator/function identity and normalized kwargs, including defaults and
  nested estimator settings; do not rely on truncated display labels.
- Actual input/output rows, columns, dtypes, array layout or sparsity, and sizes
  where cheaply available. Avoid expensive scans solely to collect metadata.
- Mode, fold, candidate membership where available, calls, processing time,
  buffer/conversion time, and relevant cache/spill counters.
- Exceptions and timeouts, with partial measurements distinguished from complete
  results. Count unsupported and failed pipelines in coverage reports.

An opaque wrapper or library call may require a second-level profile. DAG
timing identifies its boundary cost, not which internal function is responsible.
Keep nested measurements separate to avoid double counting.

Retain optimization, execution, loading, and buffer costs separately. If the
scheduler later overlaps operations, retain execution intervals and measure
critical-path impact; summed node durations will no longer represent wall time.
Synchronize asynchronous device work at the appropriate measurement boundary.

For shortlisted candidates, run at least three independent repetitions per
configuration, retain raw results, and increase repetitions if variability makes
the conclusion unclear. Interleave baseline and candidate measurements to limit
machine drift. Report the median and spread, together with paired differences.
State warm/cold I/O policy and avoid unrelated heavy jobs, including website
analysis jobs, during controlled measurements.

Treat memory numbers precisely. The current runner's sampled RSS follows the
pipeline interpreter; its `max_rss_mb` takes the maximum of self and child
high-water marks. That is not simultaneous process-tree peak RSS. Use a
process-tree measurement when the target workload spawns workers; whole-system
memory also includes unrelated processes.

### 4. Aggregate by operation and recurring pattern

Build rankings at three levels:

1. **Operation family:** datetime parsing, joins, encoding, model fitting,
   prediction, conversion, buffer work, optimization.
2. **Implementation and configuration:** model/transformer family, kwargs
   region, input-size/dtype regime, and phase. This defines useful kernel scope.
3. **Recurring connected subgraphs:** repeated parsing/extraction, impute and
   encode sequences, conversion chains, or feature preparation shared by several
   models. This identifies fusion, pushdown, and reuse candidates.

Store exact configurations, then aggregate common regimes. Grouping by every
parameter separately can fragment the evidence; grouping only by class hides
why costs differ. Separate fit and inference wherever possible.

For each group, report total affected stratum time, calls, time per call,
per-pipeline distribution, dataset/run coverage, and input-size distribution.
Keep absolute seconds alongside within-pipeline shares. Summed corpus cost
already accounts for frequency; do not multiply it by call count again.

Use two views of importance: actual observed search cost and a view with equal
weight per dataset or run. Declare weights rather than presenting the collected
corpus as a universal distribution of future users' workloads.

### 5. Turn profiles into optimization proposals

A slow, frequent component deserves investigation. It does not establish that
rewriting it in Rust will help. CatBoost, for example, already has native C++
training code. Inspect whether the time is in the algorithm, conversion,
preparation, unnecessary repetition, or the engine's integration.

Choose the mechanism supported by the evidence: a specialized kernel, fused
transformations, fewer copies/conversions, reduced recomputation, better reuse,
implementation selection, or lower optimizer/scheduler overhead. A kernel that
accepts common kwargs needs a semantic compatibility contract and a fallback for
unsupported configurations, including dtype/null behavior and numerical tolerance.

Estimate opportunity as:

```text
expected saved seconds ≈ sum over representative evaluations of
    weight × affected stratum time × realistically removable fraction
```

Treat overlapping candidate optimizations separately: their projected savings
cannot simply be added. For sequential execution, a fraction f accelerated by s
has an ideal overall speedup of `1 / ((1 - f) + f / s)`. For example, doubling a
component responsible for 40% of total time gives about 1.25× overall, before
new overhead. Validate the estimate with complete pipeline/search execution.

Every proposal should identify supporting pipelines and data regimes, total
cost and breadth, the proposed mechanism, expected saved seconds, implementation
scope/effort, correctness checks, and end-to-end before/after measurements.

## Using the existing tooling now

For initial discovery, open a run, select its canonical DataOps source, and use
**Actions → Runtime sweep** on an existing valid sample. Use **check what would
run**, then inspect **Runtime profile**, the operator explorer, and the runtime
store. Exclude stale or incompatible measurements from any combined analysis.

The CLI supports targeting a small pilot. This example uses an existing corpus
source; it requires the sample's data to be available and the pipeline's reads
to resolve to that sample. Run from the repository root:

```bash
uv run python -m pipeline_analyzer.runtime \
  --pipelines tab_playground_dec_21/mle_star/skrubify_openai \
  --pipelines tab_playground_dec_21/mle_star/skrubify_openai/ensemble \
  --run-in tab_playground_dec_21/sample_100k \
  --only init_code_1 \
  --out tab_playground_dec_21/mle_star/runtime_stats_skrubify_openai_sample_100k_pilot.json \
  --list
```

Remove `--list` to execute it. Use distinct output stores for experiment variants
and repetitions. For an elapsed-time validation run, select a new `--out` and add
`--no-stats`; changing this flag alone does not invalidate an existing entry.
Disable optional diagnostics too. At the inspected pin, `--no-stats` disables the
operator timing list and statistics reporting, but scheduler timestamp calls and
buffer accounting remain. Describe this as profiling disabled under the normal
runtime, rather than claiming that all timing overhead has been removed.
The current CLI discovers Python files broadly; use explicit `--only` pipeline
names to avoid executing shared helper or exploration modules. The website
normally selects names from the run's registered pipeline list.

One existing store illustrates why timings matter more than frequency counts:
`nyc_taxi_fare/meta_mle_sol_run_2/runtime_stats_pipelines_fulldata.json`, pipeline
`pipeline_01`, records approximately 171 s in datetime conversion, 100 s in a
remote CSV read, and 44 s in `LibXGBoost`. These are historical measurements on
the recorded setup, not new results or a corpus-wide ranking. They suggest
investigating parsing and I/O as well as model training.

## Running on the existing HPC setup

The adapter is implemented under [hpc/](hpc/README.md), including a build
definition, pilot campaign, Slurm worker, transfer helpers, and smoke test.
The sections below explain the design; that README has executable instructions.

The maintainer's `~/mle-star-stratum/hpc/` provides a suitable deployment
pattern: Slurm arrays, a prebuilt Singularity image, a checkout bind-mounted
from BeeGFS scratch, configurable resource requests, and push/fetch scripts.
Adapt that pattern for this corpus. The runtime CLI already works without the
website; there is no need to run a web server on the cluster.

The maintainer reports that compute nodes have internet access, as used by
previous web/API-connected agent runs. This makes remote-read pipelines
plausible, but verify access and reader authentication for their actual data
sources separately. Benchmark execution itself needs no Gemini API key and does
not run an MLE agent.

### Environment and deployment

Build a separate image from this repository's `pyproject.toml` and `uv.lock`.
The MLE-STAR image uses Python 3.12; this project requires Python >=3.14 and has
its own stratum pin and dependencies. Reuse the build pattern, including Rust
for the pinned source build, rather than assuming the existing `.sif` is an
equivalent environment. Build for the compute-node architecture and record the
image checksum alongside the benchmark manifest.

Keep code bind-mounted so measurement-tool changes do not require rebuilding
the image unless dependencies change. If following the existing image's
`uv sync --no-install-project` approach, expose the bound checkout's `tools/`
directory on `PYTHONPATH` inside the container; that is where
`pipeline_analyzer` and `dataset_sample` live. Use the image's Python directly
for execution, without attempting a dependency sync in every array task.

Use a separate cluster project directory for this corpus. Adapt push/fetch
filters to preserve remote data, generated runtime stores, memory traces, and
benchmark campaign artifacts. Do not copy the other project's deletion rules
unchanged: this corpus stores outputs next to run sources, rather than in one
agent workspace directory. Keep experiment outputs in unique paths and fetch
them without overwriting another campaign's measurements.

### First allocation and smoke test

Start with **one array task, 16 CPUs, 60G memory**, matching the resource request
in the MLE-STAR configuration template. This is an initial pilot allocation,
not a claim that every full-data pipeline fits. Run one known pipeline on a
validated sample, check its score, data location, runtime, and memory, and then
size larger jobs from those observations. Allow time for container startup and
result writing beyond the runner's per-pipeline timeout.

Run through Slurm on a compute node, with explicit CPU binding and recorded
thread settings. Common BLAS/OpenMP environment variables help, but are not a
complete limit: models can have their own `thread_count`, `n_jobs`, or device
settings. Verify those effective settings and their compatibility with the
allocation. For controlled comparisons, prefer exclusive-node allocations
where available; merely requesting one node does not establish exclusivity.

Begin with concurrency one. Later parallelize independent benchmark cells
across separate nodes, recording hardware identity and avoiding concurrent
measurements that contend on the same node. Shared storage or network traffic
can still affect simultaneous jobs on different nodes, so record the I/O
conditions and validate shortlisted improvements under controlled conditions.

### Array design and outputs

Use a manifest whose cells identify dataset, run, canonical source, selected
pipeline(s), data version, configuration, and repetition. Initially one cell
can run a small sequential subset through `pipeline_analyzer.runtime`. Larger
campaigns can split subsets by estimated runtime to avoid one long pipeline
holding up an entire run. The existing runner creates a new child process for
each pipeline within that subset.

Give every array cell its own `--out` store and artifact directory. Multiple
jobs must not write one store: atomic file replacement does not merge
concurrent updates. Retain Slurm job/array IDs, hostname, allocation, logs,
image identity, and the experiment manifest with the results. Keep submissions'
manifests immutable, as the MLE-STAR submit script already does.

Use node-local storage for staged data and temporary files when the aim is
compute performance; retain outputs on shared storage or copy them there
before exit. Staging must preserve each pipeline's path semantics. A fixed
`gs://` read still accesses GCS unless the benchmark explicitly adapts its
input binding and records that adaptation. Keep remote-I/O and staged-local
experiments separate.

The current website discovers stores beside their dataset/run. To browse HPC
results, fetch or publish selected stores there with matching source/data
identities. Per-cell stores can appear separately; combining them into a
validated cohort requires an explicit aggregation step that preserves
repetitions and provenance. Website sweep locking does not coordinate jobs
submitted directly through Slurm.

### Scope of the adapter

The implemented adapter contains a project-specific container definition,
configurable push/fetch helpers, an immutable campaign manifest generator, a
Slurm worker invoking the runtime CLI, and a submit script controlling resources
and concurrency. Measurement semantics remain in `pipeline_analyzer`.

This is sufficient for the current stratum profiling workflow. An array
dimension named `engine=skrub` will only become meaningful after implementing
the isolated baseline path described above. The other project's `skrub` versus
`baseline` agent modes are a different experiment and should not be reused as
if they selected this runner's execution engine.

## Recommended implementation order

1. **Establish reliable experiment identity and semantics.** Add complete
   per-measurement configuration/provenance, helper/data identity, retained
   repetitions, and filtering of stale/incompatible entries. Update scoring
   diagnostics and represent multiple scoring calls explicitly. Verify a small
   cohort against plain skrub before expanding it.
2. **Connect stratum diagnostics to the existing runner.** Retain execution plan
   snapshots and node/fold/mode timings, link them from stores, and add estimator
   kwargs and actual input metadata. Version the schema and keep raw artifacts.
3. **Add comparisons and corpus aggregation.** Provide an isolated skrub engine
   path, profiling controls, and cross-run rankings
   with dataset/run coverage and declared weights. Extend the existing website
   to browse this evidence.
4. **Run the representative pilot, then expand.** Validate shortlisted targets
   at larger sizes. Keep failure/timeout coverage visible; a successful-only
   ranking can omit the hardest workloads.
5. **Add search replay and validate improvements.** Preserve the intended
   candidate availability and session lifetime, compare reuse policies, and
   demonstrate an end-to-end saving on workloads beyond the examples used to
   design the optimization.

The first useful result is a short list of costly, recurring operations with
enough attribution to choose an optimization. The final evidence must show
correctness and elapsed-time improvement on representative pipeline or search
workloads, with scope and limitations recorded.

## Implementation references

- [Website setup and runtime sweep workflow](website/README.md)
- [Website sweep command construction](website/backend/actions.py)
- [Website runtime aggregation](website/backend/runtime_profile.py)
- [Runtime runner and freshness rules](tools/pipeline_analyzer/runtime.py)
- [Measurement child and scoring interception](tools/pipeline_analyzer/_measure.py)
- [Static DAG extraction](tools/pipeline_analyzer/loader.py)
- [Analyzer usage and runtime store documentation](tools/pipeline_analyzer/README.md)
- [DataOps evaluation semantics and pitfalls](tools/skrub_dataops_summary.md)
- [Pinned stratum source](https://github.com/deem-data/stratum/tree/4c56c54b35bf43b19be2e5128230e2d9f1ef521c),
  particularly `_api.py`, `_profiler/`, `runtime/_scheduler.py`, and
  `optimizer/logical/_scoring.py`.
- [CatBoost native training source](https://github.com/catboost/catboost/blob/master/catboost/libs/train_lib/train_model.cpp)
