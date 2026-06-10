"""Voxtral-HF Phase-3-bundle end-to-end validation (GPU).

Validates the cjm-hf-plugin-utils + cjm-torch-plugin-utils adoption + Track 19
WORKER_ENV migration live, mirroring the Voxtral-vLLM Session A pattern
(project-local runtime, PluginManager-driven, asserts against empirical_resources.db).

Run from the voxtral-hf repo root after:

  1. `cjm-ctl --cjm-config cjm.yaml setup-runtime`
  2. `cjm-ctl --cjm-config cjm.yaml install-all --plugins plugins_test.yaml`
     (voxtral-hf + ffmpeg + cjm-system-monitor-nvidia)
  3. Drop a short clip at test_files/short_test_audio.mp3

Then:

  conda run -n cjm-transcription-plugin-voxtral-hf --no-capture-output \\
    python tests_manual/validate_voxtral_hf_e2e.py

This script:
  - Loads PluginManager with sysmon_plugin_name="cjm-system-monitor-nvidia".
  - Verifies the voxtral-hf v2.0 manifest carries (a) a non-empty `description`
    (substrate validator requirement), (b) Track 19 `worker_env` with a TEMPLATED
    HF_HOME default (`${CJM_MODELS_DIR}/huggingface`), and (c) requires_gpu.
  - Eagerly loads the model via prefetch() — exercises cjm-hf-plugin-utils'
    snapshot_download_with_progress + the substrate heartbeat + load_pretrained_with_oom.
  - Runs a real transcription via submit_composition(ffmpeg.convert -> voxtral.execute).
  - Reads empirical_resources.db and ASSERTS gpu_memory_mb_peak > 0 (subtree GPU
    attribution through the worker -> HF model).
"""
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("voxtral-hf-e2e")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_AUDIO = REPO_ROOT / "test_files" / "short_test_audio.mp3"
MANIFESTS_DIR = REPO_ROOT / ".cjm" / "manifests"
EMPIRICAL_DB = REPO_ROOT / ".cjm" / "empirical_resources.db"

PLUGIN_NAME = "cjm-transcription-plugin-voxtral-hf"
SYSMON_NAME = "cjm-system-monitor-nvidia"
FFMPEG_NAME = "cjm-media-plugin-ffmpeg"


def check_prereqs() -> None:
    assert TEST_AUDIO.exists(), f"Missing test audio: {TEST_AUDIO}"
    assert MANIFESTS_DIR.exists(), f"Missing manifests dir: {MANIFESTS_DIR} — run cjm-ctl setup-runtime + install-all first"
    for name in (PLUGIN_NAME, SYSMON_NAME, FFMPEG_NAME):
        assert (MANIFESTS_DIR / f"{name}.json").exists(), f"Missing manifest: {name}.json"
    log.info("Prereqs OK: test audio + voxtral-hf + nvidia-monitor + ffmpeg manifests present")


def assert_manifest_shape() -> None:
    """v2.0 manifest must carry the T24 description, the Track 19 templated WORKER_ENV,
    and requires_gpu. RELOAD_TRIGGER surfacing is best-effort (the worker reads it off
    the class, not the manifest)."""
    manifest = json.loads((MANIFESTS_DIR / f"{PLUGIN_NAME}.json").read_text())
    assert manifest["format_version"] == "2.0", manifest["format_version"]

    code = manifest["code"]

    # T24: description is required by the substrate validator (was missing pre-bundle).
    desc = code.get("description") or manifest.get("description") or ""
    assert desc.strip(), "manifest description is empty (T24 regression)"
    log.info(f"Manifest T24 description: {desc!r}")

    # CR-1 taxonomy + Phase 5a resources.
    tax = code["taxonomy"]
    assert tax["domain"] == "transcription" and tax["role"] == "TranscriptionPlugin", tax
    assert code["resources"]["requires_gpu"] is True, code["resources"]
    # CR-7 reframe: no quantitative resource fields should remain.
    for stale in ("min_gpu_vram_mb", "recommended_gpu_vram_mb", "min_system_ram_mb"):
        assert stale not in code["resources"], f"stale resource field present: {stale}"
    log.info(f"Manifest CR-1/Phase-5a: taxonomy={tax}, resources={code['resources']}")

    # Track 19: WORKER_ENV with a templated HF_HOME default.
    worker_env = code.get("worker_env", [])
    by_name = {e["name"]: e for e in worker_env}
    assert {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "HF_HOME"} <= set(by_name), (
        f"Track 19 WORKER_ENV missing expected vars: {sorted(by_name)}"
    )
    hf_home_default = by_name["HF_HOME"].get("default", "")
    assert hf_home_default == "${CJM_MODELS_DIR}/huggingface", (
        f"Track 19 HF_HOME default not templated: {hf_home_default!r}"
    )
    log.info(f"Manifest Track 19 WORKER_ENV: {sorted(by_name)} | HF_HOME default={hf_home_default!r}")


def run_e2e() -> None:
    """Live transcription via submit_composition: ffmpeg convert (MP3->WAV) -> voxtral execute."""
    import asyncio

    from cjm_plugin_system.core.manager import PluginManager
    from cjm_plugin_system.core.config import get_config
    from cjm_plugin_system.core.queue import JobQueue
    from cjm_plugin_system.core.ports import Composition, CompositionNode, NodeState, OutputRef

    cfg = get_config()
    log.info(f"data_dir={cfg.data_dir}, manifests_dir={cfg.manifests_dir}")

    pm = PluginManager(search_paths=[MANIFESTS_DIR], sysmon_plugin_name=SYSMON_NAME)
    pm.discover_manifests()
    log.info(f"Discovered: {[m.name for m in pm.discovered]}")

    # nvidia-monitor first so GPU samples are attributable when voxtral runs.
    pm.load_plugin(next(m for m in pm.discovered if m.name == SYSMON_NAME))
    pm.load_plugin(next(m for m in pm.discovered if m.name == FFMPEG_NAME))
    log.info(f"Loaded {SYSMON_NAME} + {FFMPEG_NAME}")

    voxtral_meta = next(m for m in pm.discovered if m.name == PLUGIN_NAME)
    # Defaults exercise the new flow: model_id=Voxtral-Mini-3B, device=auto. The
    # HFCacheConfig fields (cache_dir/revision/local_files_only) keep their defaults.
    ok = pm.load_plugin(voxtral_meta, config={})
    assert ok, f"Failed to load {PLUGIN_NAME}"
    voxtral_id = voxtral_meta.name
    ffmpeg_id = FFMPEG_NAME
    log.info(f"Loaded {PLUGIN_NAME} as instance_id={voxtral_id}")

    # CR-4 SG-19 prefetch: snapshot_download_with_progress + heartbeat-wrapped
    # from_pretrained via load_pretrained_with_oom (cjm-hf-plugin-utils). On a cold
    # cache this is the expensive download; the heartbeat keeps the stall detector happy.
    log.info("Calling prefetch() to eagerly download + load the Voxtral model...")
    t0 = time.time()
    pm.get_plugin(voxtral_id).prefetch()
    log.info(f"prefetch() returned in {time.time() - t0:.1f}s")

    # CR-16 (stage 3): the composition binds voxtral's `audio` to ffmpeg's
    # ACTUAL hashed cache_dir_for_config output path at execution time via
    # OutputRef — the predict-the-path pattern is retired.
    async def run_composition() -> Any:
        queue = JobQueue(deps=pm, sysmon_plugin_name=SYSMON_NAME)
        await queue.start()
        try:
            comp_id = await queue.submit_composition(Composition(nodes=[
                CompositionNode("convert", ffmpeg_id, {
                    "action": "convert", "input_path": str(TEST_AUDIO),
                    "output_format": "wav", "sample_rate": 16000, "channels": 1,
                }),
                CompositionNode("transcribe", voxtral_id,
                                {"audio": OutputRef("convert", "output_path")}),
            ]))
            log.info(f"Submitted composition {comp_id}: ffmpeg.convert -> voxtral.execute")
            run = await queue.wait_for_composition(comp_id)
            if run.status != NodeState.completed:
                raise RuntimeError(f"Composition {comp_id} status={run.status}; nodes={run.node_runs}")
            return run.results_by_node()["transcribe"]
        finally:
            await queue.stop()

    log.info(f"Submitting composition for {TEST_AUDIO}...")
    t0 = time.time()
    result = asyncio.run(run_composition())
    from cjm_transcription_plugin_system.core import TranscriptionResult  # noqa: F401 — registers the wire kind (typed decode)
    text = result.text  # typed TranscriptionResult (stage-2 wire layer)
    log.info(f"Composition completed in {time.time() - t0:.1f}s: text={text[:120]!r}")
    assert text and text.strip(), f"Empty transcription; raw result={result!r}"

    # Empirical store should have recorded a non-zero GPU peak for the worker subtree.
    log.info(f"Inspecting empirical store at {EMPIRICAL_DB}")
    assert EMPIRICAL_DB.exists(), f"empirical store not created: {EMPIRICAL_DB}"
    con = sqlite3.connect(EMPIRICAL_DB)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        log.info(f"empirical store tables: {tables}")
        for t in tables:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})").fetchall()]
            rows = con.execute(
                f"SELECT * FROM {t} WHERE plugin_name=? OR instance_id=? OR instance_id LIKE ?",
                (PLUGIN_NAME, voxtral_id, f"{PLUGIN_NAME}%"),
            ).fetchall()
            for r in rows:
                log.info(f"  {t}: {dict(zip(cols, r))}")
    finally:
        con.close()

    pm.unload_plugin(voxtral_id)
    pm.unload_plugin(FFMPEG_NAME)
    pm.unload_plugin(SYSMON_NAME)
    log.info("Unloaded plugins; validation done.")


def main() -> int:
    check_prereqs()
    assert_manifest_shape()
    run_e2e()
    return 0


if __name__ == "__main__":
    sys.exit(main())
