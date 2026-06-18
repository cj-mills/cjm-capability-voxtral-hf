# Tombstone — `validate_voxtral_hf_e2e.py` (RETIRED 2026-06-18, stage 9)

**Origin:** `cjm-transcription-plugin-voxtral-hf/tests_manual/validate_voxtral_hf_e2e.py` (Phase-3-bundle era).
**Retired because:** imported `TranscriptionResult` from the now-dissolved `cjm-transcription-plugin-system.core` shim (GitHub-archived 2026-06-18; DTO now in `cjm-capability-primitives`, registered as the wire kind there). Per the stage-9 decision the pre-overhaul `tests_manual` cohort is **retired, not patched**.

**What it validated:** cjm-hf-plugin-utils + cjm-torch-plugin-utils adoption + WORKER_ENV migration; v2.0 manifest carries non-empty `description` + templated `HF_HOME` default + `requires_gpu`; `prefetch()` eager model load (snapshot_download + heartbeat + load_pretrained_with_oom); a real transcription via `submit_composition(ffmpeg.convert → voxtral.execute)`; and `empirical_resources.db` `gpu_memory_mb_peak > 0` (subtree GPU attribution).

**Coverage status:** SUPERSEDED — `cjm-transcription-core`'s both-transcriber e2e validates voxtral end-to-end (and the {whisper,voxtral} multi-capability discovery milestone); schema-v2 validation covers the manifest.

**Reimplementation target:** none required (cores are the standing harness).
