# HPC benchmarks: Slurm and Singularity

Run the existing stratum profiler in independent Slurm array cells, following
`~/mle-star-stratum/hpc/`'s deployment pattern. No MLE agent, Gemini key, or
website process is needed. See [the methodology](../BENCHMARKING.md) for how
to interpret and prioritize results. Plain skrub comparisons and persistent
trajectory replay are not implemented by this adapter.

## Configure and build

```bash
cp hpc/config.env.example hpc/config.env
# Set CLUSTER_PROJECT_DIR and review partition/resources. This file is gitignored.
```

The maintainer's local config has already been initialized from the MLE-STAR
setup, with a separate `mle-trajectories` directory. Defaults: partition
`standard`, 16 CPUs, 60G memory, two hours, concurrency one, and an exclusive
node. These are pilot defaults; adjust after checking site policy and observed
memory/runtime. `EXCLUSIVE=0` permits shared-node allocations. Inline environment
overrides take precedence with the supplied configuration style.

Build on a Linux machine matching the compute-node architecture, with
Singularity/Apptainer, network access, and sufficient disk/memory. There is no
container engine on the development host where this adapter was written. Use
a compute allocation if your site's build policy requires it:

```bash
module load singularity # where your site uses modules
bash hpc/build.sh
bash hpc/exec.sh python hpc/campaign.py smoke
```

`build.sh` defaults to `--fakeroot`; extra arguments replace that default, e.g.
`bash hpc/build.sh --remote` on a site configured for remote builds.
`HPC_CONTAINER=apptainer` selects Apptainer. The image uses Python 3.14, Rust,
native libraries, and this repo's frozen `uv.lock`. Source is bind-mounted;
`PYTHONPATH` exposes the checkout and `tools/`. Dependencies are not reinstalled
per task. The lock includes CUDA-enabled torch on Linux x86-64, so the image
will be large even for these CPU jobs; workers do not request GPUs or use `--nv`.
The build sandbox needs tens of gigabytes; `build.sh` moves it off a small `/tmp` tmpfs.

The synthetic smoke creates 60 rows, scores a linear model with three-fold CV,
and exercises the real worker, runtime store, memory sampler, and native node
diagnostics. It needs no corpus data or network. Artifacts are retained under
`hpc/_run/smoke-*/`. To run it in an allocation:

```bash
source hpc/env.sh
srun --partition="$PARTITION" --nodes=1 --ntasks=1 --cpus-per-task="$WORKER_CPUS" \
  --mem="$WORKER_MEM" --time=00:15:00 --cpu-bind=cores \
  bash hpc/exec.sh python hpc/campaign.py smoke
```

Builds never replace an existing image. When dependencies change, build a new
filename with `SIF="$PWD/hpc/benchmarks-v2.sif" bash hpc/build.sh` and set the
same `SIF` for submission. A `.sha256` sidecar is written after building.

## Deploy

From the development machine:

```bash
bash hpc/push.sh --dry-run
bash hpc/push.sh
bash hpc/push.sh --image # only when an image was built locally
```

Push uses rsync >=3.2.3's `--mkpath` and never deletes remote files. It excludes
data, runtime outputs, caches, local config, and credentials. Configure
`hpc/config.env` separately on the cluster. Build the image there if it was not
transferred. Removed source files must be removed remotely explicitly, or use
a fresh directory; ordinary pushes do not remove them.

Stage the pilot's `tab_playground_dec_21/sample_100k/input/` and
`sample_manifest.json` separately. If that sample already exists locally:

```bash
source hpc/env.sh
rsync -avL --mkpath tab_playground_dec_21/sample_100k/ \
  "${SSH_HOST}:${CLUSTER_PROJECT_DIR}/tab_playground_dec_21/sample_100k/"
```

Check that input bytes exist before transferring; the tracked manifest alone
does not contain the data. Input symlinks must resolve on the cluster. Set
`HPC_EXTRA_BINDS=("/path/to/datasets")` in config for targets outside the checkout.

For fixed GCS reads, use `data_access="remote"`, `data="input"`, and a versioned
object/snapshot identity in the campaign. Set `GCS_CREDENTIALS` to an authorized
reader ADC JSON file on the cluster when needed. It is bound read-only and is
not baked into the image. The wrapper uses `--cleanenv --no-home`; it does not
forward your whole host environment or load the other project's secrets.
Internet/API connectivity does not itself establish bucket access. Selecting a
sample working directory never redirects a fixed remote read.

## Review and submit

The tracked [pilot.toml](pilot.toml) selects one registered historical pipeline.
Review its data version before execution:

```bash
# Development host: no Slurm, image, or data required for a draft plan.
bash hpc/submit.sh hpc/pilot.toml --dry-run

# Cluster checkout: after preparing image and data.
bash hpc/submit.sh hpc/pilot.toml
```

Dry-run planning needs Python >=3.11 (`HPC_PYTHON` can select it). Real submission
plans inside the image. Every submission gets an immutable manifest and a new
directory. Draft manifests without an image identity cannot execute; re-plan
on the cluster. The full sbatch command and job ID are recorded. Monitor with
`squeue --me` and the campaign's `logs/` directory.

Resource overrides:

```bash
WORKER_CPUS=8 WORKER_MEM=32G TIME_WORKER=01:00:00 CONCURRENCY=1 \
  bash hpc/submit.sh hpc/pilot.toml
```

The worker uses `srun --cpu-bind=cores` and explicit BLAS/OpenMP, Rust, and joblib
CPU settings. These do not rewrite a model's explicit thread/device kwargs.
The buffer pool receives 70% of the smaller of host RAM, Slurm allocation, and
detected cgroup limit. This is a pool budget, not a bound on total process RSS.

## Larger campaigns

Copy the pilot TOML and add `[[workloads]]` tables. Top-level options:
`name`, `repetitions`, `modes`, `timeout_s`, `diagnostics`, and
`implementation_selector` (`default` or `greedy`). Each workload has `run`,
optional `source` (otherwise the registry default), `data`, `data_access`,
`data_version`, optional `only`, optional string `scoring`, and `stage_data`.
Omitting `only` selects available registered lineage pipelines, excluding helper
modules. Without lineage, supply explicit pipeline names. Scoring follows the
existing runtime runner; avoid overriding a declared scorer unintentionally.

- `profile` mode collects operator statistics and, by default, native plan/node/fold diagnostics.
- `wall` mode disables stats and diagnostics; normal scheduler timing and the existing memory sampler remain enabled.

Three repetitions and both modes create six independent cells per pipeline.
Every cell in a submission has the same resource allocation; use separate
campaigns for different budgets. Split campaigns if the site's array-size limit
requires it. Concurrent jobs can still contend on shared storage/network.

Each cell has a private working directory so relative outputs such as
`catboost_info/` do not collide. With staging off, its `input/` links to the
canonical input directory. Only `input/` and the optional sample manifest are
exposed there; pipelines requiring additional relative assets need explicit
support before inclusion.

`stage_data=true` copies local `input/`, following symlinks, into per-cell
temporary storage before timing. It uses `SLURM_TMPDIR` when supplied, otherwise
the container's `/tmp`; check capacity and site storage policy. Staging does not
rewrite absolute or remote reads. Startup, hashing, inventory, and staging are
outside scored-grid-search timing.

## Outputs, retries, and the website

```text
hpc/_run/<submission>/
    campaign.toml, manifest.json, task_count, job_id
    logs/<Slurm logs>
    cells/00000/
        attempt-001.json, runtime.log, runtime.json
        runtime.mem/             full memory CSV
        diagnostics/             native stratum exports for profile mode
        completed.json           success marker
```

The manifest hashes pipeline/shared run code, runtime/HPC code, dataset/run
TOML, and dependency files. Workers reject code drift, mismatched container
dependency files, and changed image size/mtime. The image checksum is recorded
at planning; workers avoid rehashing a multi-GB image per cell. Each attempt
records allocation, CPU affinity, hardware, package versions, command, outcome,
local input paths/sizes/mtimes, and sample-manifest hash where available.
`data_version` is caller-supplied; this is not full data-content hashing and
remote bytes are not inventoried. Use versioned inputs.

Locks prevent simultaneous writers to one cell. Completed cells are retained;
incomplete/failed cells are remeasured with `--force`. To retry one cell in the
same unchanged campaign, on the cluster:

```bash
source hpc/env.sh
export HPC_MANIFEST="$PWD/hpc/_run/SUBMISSION/manifest.json"
sbatch --chdir="$PWD" --array=0 --nodes=1 --ntasks=1 \
  --partition="$PARTITION" --cpus-per-task="$WORKER_CPUS" --mem="$WORKER_MEM" \
  --time="$TIME_WORKER" --exclusive --export=ALL \
  --output="$PWD/hpc/_run/SUBMISSION/logs/retry-%A_%a.out" \
  --error="$PWD/hpc/_run/SUBMISSION/logs/retry-%A_%a.err" hpc/run_cell.sbatch
```

Keep allocation and image settings consistent. Start a new campaign for code
or config changes. Slurm kills leave logs/partial files; only cells with
`completed.json` can be published.

Retrieve and publish on the development machine:

```bash
bash hpc/fetch.sh --dry-run
bash hpc/fetch.sh
.venv/bin/python hpc/campaign.py publish hpc/_run/SUBMISSION/manifest.json
```

Fetch updates raw campaign files, never deletes files, and preserves newer
local files. Publication creates distinct
`<dataset>/<run>/runtime_stats_hpc_<manifest-hash>_<cell>.json` stores; repetitions
are not pooled and different published results are never overwritten. The
website discovers them automatically, including samples identified by a tracked
manifest without local input bytes. Keep the raw campaign and diagnostics for
drill-down. Website sweep locking does not coordinate direct Slurm submissions.

## Local checks

```bash
.venv/bin/python -m unittest discover -s hpc/tests -v
bash -n hpc/*.sh hpc/*.sbatch
```

The synthetic smoke tests both profile and wall modes through the real worker. The image
and Slurm launch still need the container smoke on the target site.

References: [Slurm sbatch](https://slurm.schedmd.com/sbatch.html),
[Singularity mounts](https://docs.sylabs.io/guides/latest/user-guide/bind_paths_and_mounts.html),
and [Singularity environment](https://docs.sylabs.io/guides/latest/user-guide/environment_and_metadata.html).
