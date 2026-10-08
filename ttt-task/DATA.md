# ttt-task — where the data comes from

**TrackTheTrackers**, not a Kaggle competition, so there is no `get_data.sh`
here. The task description says so explicitly: "In contrast to classical
competitions like Kaggle, you are not given a dataset with features for training
and prediction. Instead you are only given the targets that you need to predict
for and a 'pile of data'."

All three `mlevolve_run_*/config.yaml` point at
`/home/estrauss-ldap/repos/mle-star/machine_learning_engineering/tasks/trackthetrackers-task`,
with `exp_id: trackthetrackers-task`.

## Expected layout

Pipelines read bare `input/...` relative to their working directory, so
`--run-in ttt-task` needs all seven files:

```
ttt-task/input/
    tracking_graph_train.parquet   known tracker presence (the labelled side)
    target.tsv                     50,000 domains to predict for
    trackers.tsv                   the 355 candidate trackers + metadata
    domains.parquet                hostname lookup, 46M domain ids
    link-graph.parquet             623M hyperlink edges between domains
    url-classification.csv         content category for a subset of URLs
    freedom-of-the-press.csv       country press-freedom score, joins by TLD
```

Task: for each domain in `target.tsv`, predict up to 10 tracker ids. Metric
**Recall@10**, averaged over domains. Submission is a TSV of
`domain_id, tracking_domain_id` pairs; rows past the tenth per domain are
ignored, and a domain with no predicted rows scores 0.

`tracker_id` is a compact 0–354 index, which is why pipelines hardcode
`NUM_TRACKERS = 355`. Any subset of this data must keep all 355 tracker rows.

## Full data on this machine

Verified cloud inputs: `gs://mle-trajectories-data/ttt-task/input`.
See [CLOUD.md](CLOUD.md) for direct reads and private scoring locations.

The canonical local copy is under `~/datasets`, with seven inputs totalling
3,884,913,139 bytes:

```
~/datasets/trackthetrackers-task/data/          + TASK.md, scoring/score.py,
                                                  scoring/target_with_labels.tsv
```

Note the first nests the files under `data/`, not `input/`, so link or copy the
*contents*. `ttt-task/input/` is **already populated on this machine** with
symlinks to `~/datasets/trackthetrackers-task/data/`:

```bash
mkdir -p ttt-task/input
for name in tracking_graph_train.parquet target.tsv trackers.tsv domains.parquet \
    link-graph.parquet url-classification.csv freedom-of-the-press.csv; do
    ln -sfn "$HOME/datasets/trackthetrackers-task/data/$name" "ttt-task/input/$name"
done
```

`pipeline_analyzer.runtime` checks for dangling symlinks under `input/` and warns
before spending a sweep discovering them
(`tools/pipeline_analyzer/runtime.py:398-403`), so if that dataset directory
moves, the breakage is reported rather than hit at read time.

The original scorer and held-out labels remain outside the repository under
`~/datasets/trackthetrackers-task/scoring/`; their cloud copies are maintainer-only
at `gs://mle-trajectories-private/ttt-task/scoring/`.

## Sampling

This dataset already has the worked sampler: **`make_input_sample.py`**, which is
the reference example for the `dataset-sample` skill.

```bash
python ttt-task/make_input_sample.py      # 20k domains -> ttt-task/sample/input/
TTT_N_SEED=200000 TTT_N_DOMAINS=200000 TTT_N_TARGET=5000 \
    TTT_OUT=ttt-task/sample_200k/input python ttt-task/make_input_sample.py
```

Each size is its own run-root, so it is one `--run-in` away:

| run-root | domains.parquet rows | size |
|---|---|---|
| `ttt-task` | 46M (full, symlinked) | ~3.8 GB |
| `ttt-task/sample` | 22,350 | 1.9 MB |
| `ttt-task/sample_200k` | 205,339 | 12 MB |
| `ttt-task/sample_1m` | 1,020,327 | 55 MB |
| `ttt-task/sample_5m` | 5,020,274 | 324 MB |

(`target.tsv` tops out at its own 20,000 rows from `sample_1m` upward.)

Why it is not a row sample, in its own words: "Random domain sampling would
destroy the link graph (623M edges over 46M nodes: two random domains are almost
never connected), so the sample is a NEIGHBOURHOOD: seed domains plus their
link-graph neighbours that are themselves tracked, then every edge whose
endpoints both survive."

The 20k-domain sample it produces is what the hand conversion in
`mlevolve_run_2/skrubify_manual/` was verified against — per-fold design matrix
bit-identical to the original (train 16000x1860, val 4000x1860) and score
`0.7805021056023915`. See `skrubify_manual/versions/README.md`.
