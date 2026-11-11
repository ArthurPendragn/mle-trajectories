"""Exercise shell entry points with fake transports; no cluster/network calls."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ShellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.capture = self.folder / "calls.json"
        self.fake = self.folder / "transport"
        self.fake.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
p = pathlib.Path(os.environ["HPC_CAPTURE"])
rows = json.loads(p.read_text()) if p.exists() else []
rows.append(sys.argv[1:])
p.write_text(json.dumps(rows))
''')
        self.fake.chmod(0o755)
        self.image = self.folder / "test.sif"
        self.image.write_bytes(b"test")
        self.env = dict(os.environ, HPC_CAPTURE=str(self.capture), HPC_CONTAINER=str(self.fake),
                        SIF=str(self.image), WORKER_CPUS="4", SLURM_CPUS_PER_TASK="6",
                        GCS_CREDENTIALS="", CLUSTER_PROJECT_DIR="/scratch/fixture",
                        SSH_HOST="unused-fake-host", PATH=f"{self.folder}:{os.environ['PATH']}")

    def test_container_uses_explicit_clean_environment_and_allocated_threads(self):
        subprocess.run(["bash", "hpc/exec.sh", "python", "hpc/campaign.py", "smoke"],
                       cwd=ROOT, env=self.env, check=True, capture_output=True)
        args = json.loads(self.capture.read_text())[0]
        self.assertIn("--cleanenv", args)
        self.assertIn("--no-home", args)
        self.assertIn("OMP_NUM_THREADS=6", args)
        self.assertIn("LOKY_MAX_CPU_COUNT=6", args)
        self.assertEqual(args[-3:], ["python", "hpc/campaign.py", "smoke"])

    def test_push_and_fetch_preserve_data_and_never_delete_remote_outputs(self):
        (self.folder / "rsync").symlink_to(self.fake)
        for script in ("push", "fetch"):
            subprocess.run(["bash", f"hpc/{script}.sh", "--dry-run"], cwd=ROOT,
                           env=self.env, check=True, capture_output=True)
        calls = json.loads(self.capture.read_text())
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("--delete" not in call and "--dry-run" in call for call in calls))
        self.assertIn("--exclude=input/", calls[0])
        self.assertIn("--exclude=/hpc/config.env", calls[0])
        self.assertIn("--exclude=runtime_stats_*", calls[0])
        self.assertIn("--update", calls[1])


if __name__ == "__main__":
    unittest.main()
