# dataset_sample

Persisted samples of a dataset, built from the recipe in its `dataset.toml`, so
a runtime sweep, skrubify's repair loop or any pipeline run can use a smaller
copy of the data **without patching anything**: the sample is a second run-root
with the same file names as `input/`.

```
<dataset>/
    input/                          full data
    sample_100k/
        input/                      same files, fewer rows (gitignored)
        sample_manifest.json        how it was built (tracked)
```

```bash
python -m dataset_sample tab_playground_dec_21 --size 100k --dry-run   # what it would do
python -m dataset_sample tab_playground_dec_21 --size 100k             # -> sample_100k/
python -m dataset_sample tab_playground_dec_21 --size 100k --force     # rebuild
```

The website's run page has the same as an action ("Build sample").

## Recipe

```toml
[sample]
target = "Cover_Type"      # stratify sampled tables on this column (optional)
min_per_class = 5          # every class keeps at least this many rows (or all it has)
seed = 0

[sample.tables]
"train.csv" = "sample"                                     # reduced to --size rows
"test.csv" = "keep"                                        # copied whole (hard link)
"sample_submission.csv" = { match = "test.csv", key = "id" }   # rows whose key survived
"orders.parquet" = { match = "train.csv", key = "customer_id" } # star schema: follow the keys
```

- `sample`: a seeded random sample of `--size` rows, proportionally stratified
  on `target` when the table has that column.
- `keep`: the whole file (hard link, so no extra space). Files the recipe does
  not list are kept whole too, and the manifest says so.
- `match`: the rows whose `key` value survived in the other table's
  `from_key` column (default: the same name). Chains are followed in order.

CSV/TSV records are copied byte for byte (quoting, formatting and column order
exactly as in the source, so pandas infers the same dtypes); parquet keeps its
schema. Tables are streamed, so a multi-GB file is fine (dec21's 4M-row
`train.csv` → 100k rows in 11 s, 335 MB peak).

A dataset whose shape needs its own reasoning (a graph, a data lake, images)
keeps a per-dataset script (see the `dataset-sample` skill) and points at it:

```toml
[sample]
script = "make_input_sample.py"
out_env = "TTT_OUT"                          # where the script writes input/
size_env = ["TTT_N_SEED", "TTT_N_DOMAINS"]   # knobs set to --size
```

## What a build guarantees

The sample is built in a hidden `.<name>.building/` folder and moved into
place only after these checks pass, so a failed build never leaves a
half-written sample:

- the same file set as `input/`, the same columns per table (and parquet schemas),
- no table emptied,
- every class of `target` still present.

`sample_manifest.json` records the recipe and its hash, the size, seed, rows
per file before and after, class counts, the checks, and a **fingerprint** (file
names and sizes under `input/`). `pipeline_analyzer.runtime` stores the
fingerprint with every entry measured on a sample, so rebuilding the sample
differently makes those entries stale. `python -c "from dataset_sample.core import
adopt; ..."` writes a manifest for a sample built before this tool (the ttt
samples).
