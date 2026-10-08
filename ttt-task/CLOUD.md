# TrackTheTrackers data on GCS

Input root: `gs://mle-trajectories-data/ttt-task/input`.
All seven original inputs are verified: 3,884,913,139 bytes total. The original
split, domain IDs, tracker IDs and labels are unchanged.

| File | Contents |
| --- | --- |
| `tracking_graph_train.parquet` | 36,674,685 known domain/tracker pairs; `domain_id`, `tracking_domain_id`, `tracker_id` |
| `target.tsv` | 50,000 held-out domain IDs to predict for |
| `trackers.tsv` | All 355 candidate trackers and their metadata; compact `tracker_id` values 0–354 |
| `domains.parquet` | 46,269,087 hostname/domain-ID lookup rows |
| `link-graph.parquet` | 623,056,313 hyperlink edges; `source_domain_id`, `target_domain_id` |
| `url-classification.csv` | URL content categories |
| `freedom-of-the-press.csv` | Country press-freedom scores, joinable by TLD |

Build features and training examples from these tables. For every target domain,
submit up to ten tracker domain IDs in a headered TSV with columns `domain_id`
and `tracking_domain_id`. Submit `tracking_domain_id`, rather than the compact
`tracker_id`. Recall@10 is averaged across all target domains; missing predictions
score zero, and only the first ten rows for each domain count.

The target domains are absent from the labelled training graph. The original
task description is published at `gs://mle-trajectories-data/ttt-task/TASK.md`;
its original `data/` references correspond to the GCS input root above.

Read directly with reader ADC:

```python
import pandas as pd

root = "gs://mle-trajectories-data/ttt-task/input"
targets = pd.read_csv(f"{root}/target.tsv", sep="\t")
trackers = pd.read_csv(f"{root}/trackers.tsv", sep="\t")
known = pd.read_parquet(
    f"{root}/tracking_graph_train.parquet",
    columns=["domain_id", "tracking_domain_id", "tracker_id"],
)
```

`cloud_manifest.json` records the verified input inventory, sizes, CRC32C
checksums and object generations. The original `score.py` and
`target_with_labels.tsv` are exclusively in
`gs://mle-trajectories-private/ttt-task/scoring/`, with a separate private
manifest at `gs://mle-trajectories-private/ttt-task/cloud_manifest.json`.
The agent reader cannot access those objects. Maintainer scoring instructions
are in [infra/README.md](../infra/README.md#trackthetrackers).

Historical local reads use the seven symlinks in `ttt-task/input/`, pointing
to `~/datasets/trackthetrackers-task/data/`. Scoring files stay outside the
repository under `~/datasets/trackthetrackers-task/scoring/`.
