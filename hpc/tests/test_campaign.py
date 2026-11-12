"""Behavioral checks for registry selection, isolation, recovery, and publication."""
import contextlib
import fcntl
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hpc import campaign


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "dataset/run"
        self.pipes = self.run / "pipelines"
        for folder in (self.pipes, self.root / "hpc", self.root / "tools/pipeline_analyzer"):
            folder.mkdir(parents=True)
        (self.root / "pyproject.toml").write_text("# dependencies\n")
        (self.root / "uv.lock").write_text("# lock\n")
        (self.root / "dataset/dataset.toml").write_text('label = "fixture"\n')
        (self.run / "run.toml").write_text('agent = "mle-claude"\n')
        for name in ("p1", "p2", "common", "data_exploration_1"):
            (self.pipes / f"{name}.py").write_text(f"# {name}\n")
        (self.pipes / "results.json").write_text(json.dumps([
            {"pipeline": "p1", "score": 0.5}, {"pipeline": "p2", "parent": "p1", "score": 0.6}]))
        self.input = self.root / "dataset/input"
        self.input.mkdir()
        (self.input / "train.csv").write_text("x,y\n1,2\n")
        self.config = self.root / "campaign.toml"
        self.image = self.root / "image.sif"
        self.image.write_bytes(b"test image")
        self.manifest = self.root / "hpc/_run/test/manifest.json"

    def plan(self, *, top="", workload=""):
        self.config.write_text('name="test"\n' + top + '\n[[workloads]]\nrun="dataset/run"\n'
                               'data_version="fixture-v1"\n' + workload)
        with contextlib.redirect_stdout(io.StringIO()):
            return campaign.plan(self.config, self.manifest, self.image, root=self.root)

    def execute(self, index=0, *, fail=False):
        def runtime(cmd, **kwargs):
            self.command = cmd
            self.runtime_env = kwargs["env"]
            self.runtime_cwd = Path(cmd[cmd.index("--run-in") + 1])
            self.assertTrue((self.runtime_cwd / "input/train.csv").is_file())
            self.assertIn("--force", cmd)
            out = Path(cmd[cmd.index("--out") + 1])
            name = cmd[cmd.index("--only") + 1]
            self.store = out
            out.write_text(json.dumps({"meta": {"run_in": str(self.runtime_cwd)}, "pipelines": {
                name: {"status": "error" if fail else "ok", "wall_s": 1.0}}}))
            return SimpleNamespace(returncode=1 if fail else 0)

        with patch.dict(os.environ, {"HPC_IMAGE": str(self.image)}), \
             patch.object(campaign.subprocess, "run", side_effect=runtime) as call, \
             patch.object(campaign, "environment_info", return_value={"hostname": "node-a"}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = campaign.run_cell(self.manifest, index, root=self.root, dependency_root=self.root)
        return result, call

    def test_selects_lineage_pipelines_not_helper_modules(self):
        document = self.plan()
        self.assertEqual([c["pipeline"] for c in document["cells"]], ["p1", "p2"])
        with self.assertRaises(FileExistsError):
            self.plan() # campaign manifests are immutable

    def test_repetitions_and_modes_are_independent_cells(self):
        document = self.plan(top='repetitions=3\nmodes=["profile","wall"]\n')
        cells = document["cells"]
        self.assertEqual(len(cells), 12)
        self.assertEqual(len({c["id"] for c in cells}), 12)
        self.assertTrue(all(c["diagnostics"] == (c["mode"] == "profile") for c in cells))

    def test_rejects_helper_as_pipeline_and_unknown_engine(self):
        with self.assertRaisesRegex(ValueError, "unregistered"):
            self.plan(workload='only=["common"]\n')
        with self.assertRaisesRegex(ValueError, "unknown campaign"):
            self.plan(top='engine="skrub"\n')
        with self.assertRaisesRegex(ValueError, "no DataOps source"):
            self.plan(workload='source="missing"\n')

    def test_remote_sample_and_remote_staging_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample cwd"):
            self.plan(workload='data="sample_10"\ndata_access="remote"\n')
        with self.assertRaisesRegex(ValueError, "requires local"):
            self.plan(workload='data_access="remote"\nstage_data=true\n')

    def test_code_and_dependency_drift_prevent_execution(self):
        self.plan(workload='only=["p1"]\n')
        (self.pipes / "common.py").write_text("# changed helper\n")
        with self.assertRaisesRegex(ValueError, "code changed"):
            self.execute()

    def test_image_drift_and_container_dependency_mismatch_are_rejected(self):
        self.plan(workload='only=["p1"]\n')
        self.image.write_bytes(b"a different image")
        with self.assertRaisesRegex(ValueError, "image changed"):
            self.execute()
        self.image.write_bytes(b"test image")
        document = json.loads(self.manifest.read_text())
        document["image"] = campaign.image_identity(self.image)
        self.manifest.write_text(json.dumps(document))
        wrong = self.root / "other-container"
        wrong.mkdir()
        (wrong / "pyproject.toml").write_text("wrong dependencies")
        with patch.dict(os.environ, {"HPC_IMAGE": str(self.image)}):
            with self.assertRaisesRegex(ValueError, "container pyproject"):
                campaign.run_cell(self.manifest, 0, root=self.root, dependency_root=wrong)

    def test_concurrent_writers_are_rejected(self):
        self.plan(workload='only=["p1"]\n')
        directory = self.manifest.parent / "cells/00000"
        directory.mkdir(parents=True)
        with (directory / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.assertRaisesRegex(ValueError, "already running"):
                self.execute()

    def test_stage_local_inputs_recover_failure_and_keep_completed_result(self):
        self.plan(workload='only=["p1"]\nstage_data=true\n')
        result, _ = self.execute(fail=True)
        self.assertEqual(result, 1)
        self.assertNotEqual(self.runtime_cwd, self.root / "dataset")
        self.assertEqual(Path(self.runtime_env["SKB_DATA_DIRECTORY"]).parent,
                         self.runtime_cwd.parent)
        self.assertFalse(self.runtime_cwd.exists()) # temporary stage cleaned up
        result, _ = self.execute()
        self.assertEqual(result, 0)
        original = self.store.read_bytes()
        result, call = self.execute()
        self.assertEqual(result, 0)
        call.assert_not_called()
        self.assertEqual(self.store.read_bytes(), original)
        self.assertEqual(len(list(self.store.parent.glob("attempt-*.json"))), 2)

    def test_missing_and_dangling_data_fail_before_runtime(self):
        self.plan(workload='only=["p1"]\n')
        (self.input / "train.csv").unlink()
        with self.assertRaisesRegex(ValueError, "no local data"):
            self.execute()
        (self.input / "train.csv").symlink_to("/does-not-exist")
        with self.assertRaisesRegex(ValueError, "dangling"):
            self.execute()

    def test_profile_and_wall_flags_and_publish_preserve_repetitions(self):
        self.plan(top='repetitions=2\nmodes=["wall"]\n', workload='only=["p1"]\n')
        self.execute(0)
        self.assertIn("--no-stats", self.command)
        self.assertNotIn("--diagnostics-dir", self.command)
        self.execute(1)
        with contextlib.redirect_stdout(io.StringIO()):
            campaign.publish(self.manifest, root=self.root)
            campaign.publish(self.manifest, root=self.root) # idempotent publication
        stores = sorted(self.run.glob("runtime_stats_hpc_*.json"))
        self.assertEqual(len(stores), 2)
        reps = {json.loads(p.read_text())["meta"]["hpc"]["cell"]["repetition"] for p in stores}
        self.assertEqual(reps, {1, 2})
        stores[0].write_text("modified local artifact")
        with self.assertRaisesRegex(ValueError, "refusing to replace"):
            campaign.publish(self.manifest, root=self.root)

    def test_website_recognizes_fetched_sample_without_local_input_bytes(self):
        from website.backend.registry import load_dataset

        sample = self.root / "dataset/sample_10"
        (sample / "input").mkdir(parents=True)
        (sample / "input/train.csv").write_text("x,y\n1,2\n")
        (sample / "sample_manifest.json").write_text('{"size":"10"}')
        self.plan(workload='only=["p1"]\ndata="sample_10"\n')
        self.execute()
        with contextlib.redirect_stdout(io.StringIO()):
            campaign.publish(self.manifest, root=self.root)
        (sample / "input/train.csv").unlink()
        (sample / "input").rmdir()
        dataset = load_dataset(self.root / "dataset")
        self.assertEqual(dataset.runs[0].runtime[0].data, "sample_10")


if __name__ == "__main__":
    unittest.main()
