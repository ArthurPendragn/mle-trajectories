# website

A browser interface over the whole corpus: every dataset and agent run, what
state each is in (lineage, skrub plans, runtime measurements, data), and —
growing from here — the analyses `tools/pipeline_analyzer` produces, across runs
rather than one static report per run.

```
website/
    run.sh              start both processes
    backend/            Python: the corpus registry + a JSON API (FastAPI)
        registry.py     reads dataset.toml / run.toml, derives everything else,
                        picks the source / runtime store a view uses
        app.py          /api/corpus, /api/runs/<dataset>/<run>[/tree|/runtime/<store>|/analysis/<source>]
        tree.py         search tree (graphviz), from the lineage alone
        runtime_profile.py  per-pipeline and per-operator time, from a runtime store
        worker.py       operator analysis of one run+source, in its own process
        jobs.py         starts workers, caches their results in website/.cache/
        actions.py      long-running actions (runtime sweep): builds the command,
                        runs it detached, tracks it in website/.cache/actions/
        __main__.py     serves the API on a private unix socket
    frontend/           Next.js (App Router, TypeScript), plain CSS
        proxy.ts        optimistic login gate
        lib/dal.ts      the only code that calls the API; re-checks the session
        lib/capabilities.ts   what each action needs, and why it is unavailable
        components/explorer/  the operator explorer (engine.js ported from
                        tools/pipeline_analyzer/explorer.js)
        components/     search tree + analyses, runtime profile, source pickers
        app/(main)/     corpus page, run page
        app/login/      login form
```

## Setup (once)

Node is not installed system-wide on the node; it lives in `~/.local/opt/node`
(the official prebuilt linux-x64 tarball, symlinked into `~/.local/bin`).

```bash
uv sync --group website                     # fastapi + uvicorn
(cd website/frontend && npm ci)
(cd website/frontend && npm run set-password)   # the login; writes .env.local (0600)
```

## Run

```bash
website/run.sh          # development: API and frontend reload on code changes
website/run.sh prod     # production build, then serve it
```

On the laptop (inside the VPN), open a tunnel and browse to
<http://localhost:3000>:

```bash
ssh -N -L 3000:localhost:3000 <node>
```

`PORT=3100 website/run.sh` changes the port (tunnel to the same one).

## Security model

- **Nothing listens on the network.** The frontend binds `127.0.0.1` only; the
  SSH tunnel is the way in. The API has no TCP port at all: it listens on a unix
  socket inside a `0700` directory (`$XDG_RUNTIME_DIR/mle-trajectories/`), so
  other users of the node cannot reach it.
- **One login**, because other users of the node *can* reach `127.0.0.1:3000`:
  scrypt-hashed password in `frontend/.env.local` (gitignored, mode 0600), a
  signed `HttpOnly`, `SameSite=Strict` session cookie (7 days; `set-password`
  rotates the secret and so signs everyone out), 1 s delay per failed attempt and
  a 5-minute lockout after 5 failures in 15 minutes.
- Session checked twice: optimistically in `proxy.ts`, and authoritatively in
  `lib/dal.ts` before every API call. Server actions (login/logout) get Next's
  built-in Origin check against cross-site requests.
- The API addresses runs only by ids the registry produced — a request is never
  joined onto a filesystem path.
- `X-Frame-Options: DENY`, `nosniff`, `no-referrer`; Next telemetry disabled by
  `run.sh`; no external fonts or CDNs.

## Manifests

The registry derives what it can from the folders — trajectory file and format,
original scripts (`pipelines/` and its sub-folders), every `skrubify*/` folder,
every `runtime_stats_*.json` (matched to the folder it measured, with how many
entries are stale) — so a new skrubify folder or runtime store appears without
editing anything. Two small files record the rest:

`<dataset>/dataset.toml`

```toml
label = "NYC taxi fare"
task = "regression"
data = "local"                 # default; or a url, e.g. "gs://bucket/lake"
note = "..."

[metric]                       # MLE-STAR and mlevolve do not record its name
name = "rmse"
lower_is_better = true

[defaults]                     # for runs with several sources / runtime stores
source = "skrubify_5_6_sol"    # preferred skrubify folder, where a run has it
runtime = "full"               # "full" | "sample": which kind of store to prefer
```

`<dataset>/<run>/run.toml`

```toml
agent = "mle-star"             # mle-star | mlevolve | mle-claude   (required)
label = "MLE-STAR run 1"
note = "shown on the run page"
# trajectory = "final_state.json"          # default: detected
# originals = ["pipelines/1"]              # default: pipelines/ + sub-folders with .py

# [metric] overrides the dataset's

[sources.skrubify_openai]      # annotate a skrubify folder (key = folder name)
label = "OpenAI translation"
default = true                 # the source analyses use by default
fold_identical_code = true     # pipeline_analyzer --fold-identical-code
# dirs = ["skrubify_openai"]   # default: the folder + sub-folders with .py
# hidden = true
note = "..."

[runtime.runtime_stats_skrubify_openai_fulldata]   # annotate a store (key = file stem)
label = "full data"
default = true
# hidden = true
note = "..."
```

### Which source and runtime store a view uses

The operator explorer and operator statistics depend on the **skrub source**,
the runtime profile on the **runtime store**. Both are picked on the run page
(kept in the URL, `?source=&runtime=`, so a view can be linked), and resolved
in this order — the page says which rule applied:

1. the page's explicit choice,
2. the only one there is,
3. `default = true` in `run.toml`,
4. `[defaults]` in `dataset.toml`,
5. built in: the source covering the most pipelines (the agent's own plans on a
   tie); the store measuring the selected source, full data before sampled,
   then the most pipelines measured.

For mle-claude runs the agent's own `pipelines/` are the skrub plans and
`pipelines/results.json` is the lineage; its entries (not a file pattern) define
which files are pipelines, so shared modules and `data_exploration_*.py` are
left out.

Check the manifests and see what the site will show, without starting it:

```bash
uv run python -m website.backend.registry                 # one line per run + warnings
uv run python -m website.backend.registry --run ttt-task/mlevolve_run_2
```

## Operator analysis: background builds and cache

Extracting operator DAGs imports every pipeline (skrub, stratum, torch, the
run's own modules), so it runs in `backend/worker.py`, never in the API. Opening
a run page starts the build for the selected source; the page polls and shows
the explorer when it is ready (5 s for the 68-pipeline dec21 MLE-STAR run, 13 s
for the nyc data-lake run). At most two builds run at once; more are queued.

Results are cached in `website/.cache/analysis/<dataset>/<run>/<source>-<key>.json`
(gitignored). The key hashes every `.py` file in the source's folders, the
lineage file, the stratum build and the analyzer's own code, so editing a
pipeline, re-pinning stratum or changing the analyzer rebuilds automatically;
nothing needs invalidating by hand. A failed build shows its log and a retry
button.

The worker turns off skrub's per-DataOp creation-stack recording (only used
for "defined here" in error messages): it was ~80% of plan-building time, and
the extracted DAGs are byte-identical without it.

## Operator explorer: grouping

Before layout, every connected region of operations carried by the same ticked
pipelines collapses into one stacked box ("12 operations · Numeric ×4 ·
Drop ×2"), recomputed on every selection change. Click a box to expand it;
click one of its operations to collapse it again from the inspector; "expand
all" / "collapse all"; "group" turns it off. Estimators stay visible and split
the regions around them unless "estimators too" is ticked. On the nyc run, all
32 pipelines' 1364 operations draw as 131 boxes.

The collapsed graph is always acyclic. An operation's inputs belong to every
pipeline the operation belongs to, so pipeline sets only shrink along a path,
and a path leaving a region and re-entering it can only pass through operations
of that same region. Estimators kept visible would break this, so regions are
also cut at each estimator (keyed by the estimators upstream of them).
`contract()` in `components/explorer/engine.js` is the pure core. It throws
rather than drawing a cycle, and was property-tested on the cached analyses:
2242 random selections, diff pairs and expand states, with no cycle, no mixed
pipeline set and no disconnected group.

## Actions: runtime sweep

The run page's **Actions** section runs `pipeline_analyzer.runtime` over one
skrub source. You choose:

- **data**: the dataset's full `input/`, or any sample folder. A dataset folder
  is recognised as data when it holds `input/`, and every other sub-folder with
  its own `input/` is a sample (`sample/`, `sample_200k/`, …; run with it as
  `--run-in`). A `sample_manifest.json` from `make_sample.py` adds row counts to
  the option.
- **row cap**: `--sample-rows N` caps every `pandas.read_csv` at N rows, on
  either kind of data folder. Parquet and other readers are not capped.
- the timeout per pipeline, retry failed, re-measure all.

The sweep writes the store that already holds this source at this data size
(data folder + row cap), which the tool treats as a cache: fresh entries are
kept, missing and stale ones (other code, other stratum build) re-measured.
Otherwise it creates `runtime_stats_<source>[_<sample>][_<N>rows|_fulldata].json`.
"check what would run" shows the tool's own `--list` (~12 s, because it imports
stratum for the build). Starting takes two clicks.

The job runs detached (it survives an API reload) with its state in
`website/.cache/actions/<id>/` (spec, log, pid, exit). The page polls it and
shows its progress and log. **stop** takes down the whole process tree; whatever
was measured so far stays in the store. Only one sweep runs at a time on the
node, because two side by side would slow each other down and skew the timings.
When a sweep ends, its store shows up in the runtime-store picker.

## Adding an analysis or action

1. For an action, declare it in `frontend/lib/capabilities.ts` with its
   requirement (lineage, skrub plans, runtime store, data, ...) — the run page
   shows it greyed out with the reason where the run cannot support it. An
   analysis section states its own reason when its input is missing.
2. Add the backend endpoint in `backend/app.py`; anything that imports skrub,
   stratum or a pipeline belongs in a worker process, not in the API process.
   A long-running action builds its command in `backend/actions.py` from the
   registry run and validated parameters (never a path or command from the
   request), and starts it with `actions._start(spec)`.
3. Add the page or component under `frontend/app/(main)/`. Mutations are
   server actions (`frontend/app/actions/`), which get Next's Origin check.

## Status

Done: registry and manifests, API, login, corpus page, run page — source and
runtime-store pickers, search tree, operator explorer, operator statistics
(logical, physical), runtime profile, sources and coverage, runtime stores
(with a quiet hint when measured under an older stratum build), trajectory
metadata, steps with Δ vs parent, runtime sweep action.

Left out on purpose: the static report's per-step diff sections (the
explorer's "diff vs parent" colouring covers one step on demand).

Next: global views across runs; more actions.
