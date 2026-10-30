# Skrubify on the cluster

Same shape as `mle-star-stratum`: a Singularity image holds the locked
dependencies, the repo is bind-mounted, and SLURM runs one array task per
family in `hpc/batches.txt`. A task converts that family in one `skrubify`
process, so `train0` and its `train0_improve*` scripts stay together.

`init_code_1` through `init_code_4` are not in the list.

## Once

```bash
cp hpc/config.env.example hpc/config.env
cp hpc/secrets.env.example hpc/secrets.env && chmod 600 hpc/secrets.env
# edit both. PROVIDER=openai needs OPENAI_API_KEY.
# PROVIDER=gemini needs GEMINI_API_KEY, or GOOGLE_API_KEY (copied across).

singularity build --fakeroot hpc/mle_trajectories.sif hpc/mle_trajectories.def

bash hpc/push.sh --mkdir --image
```

Build the image on x86-64. Later code edits are `bash hpc/push.sh` with no
image rebuild. Rebuild only when `pyproject.toml` or `uv.lock` changes.

## Submit

On the cluster:

```bash
cd /beegfs/scratch/schustet/mle-trajectories
bash hpc/submit_skrubify.sh
```

`CONCURRENCY=2` runs two families at once. A batch that already finished
(`hpc/_run/done/<name>`) is skipped. `FORCE=1` reruns it.

A score gap in the log is the intended result of removing the leak. The job
fails only when the conversion itself fails.

```bash
squeue --me
tail -f hpc/_run/logs/skrubify-*.out
```

## Fetch

On the laptop:

```bash
bash hpc/fetch.sh          # skrubify/1/ and the json reports
bash hpc/fetch.sh --logs
```
