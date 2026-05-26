"""CR-4 reconfigure-lifecycle validation for the Voxtral-HF plugin.

Contract-level (no real model load — the model is large/GPU). Exercises the
substrate's reconfigure delta path in-process with fake model + processor objects:

  1. reconfigure(model_id flip) -> RELEASE model + processor (RELOAD_TRIGGER ->
     _release_model) + RE-APPLY config (_apply_config)
  2. device is also a RELOAD_TRIGGER
  3. on_disable releases (CR-2)

Requires the substrate version with the two-phase reconfigure (CR-4). Run from the
repo root in the plugin's env:

    conda run -n cjm-transcription-plugin-voxtral-hf --no-capture-output python tests_manual/test_reconfigure.py
"""
import sys

A = {"model_id": "test/model-a", "device": "cpu", "dtype": "float32"}
B = {"model_id": "test/model-b", "device": "cpu", "dtype": "float32"}


def main() -> int:
    from cjm_transcription_plugin_voxtral_hf.plugin import VoxtralHFPlugin

    p = VoxtralHFPlugin()
    p._apply_config(A)
    assert p.config.model_id == "test/model-a" and p.device == "cpu"

    # 1) model_id trigger: release model + processor, re-apply
    p.model = object(); p.processor = object()
    p.reconfigure(A, B)
    assert p.model is None and p.processor is None, "RELOAD_TRIGGER must release model + processor"
    assert p.config.model_id == "test/model-b", "reconfigure must re-apply config (CR-4)"
    print("[1] reconfigure model_id a->b: model+processor released + applied  OK")

    # 2) device trigger
    p.model = object(); p.processor = object()
    p.reconfigure(B, {"model_id": "test/model-b", "device": "auto", "dtype": "float32"})
    assert p.model is None and p.processor is None, "device change must release"
    print("[2] reconfigure device cpu->auto: released  OK")

    # 3) on_disable releases (CR-2)
    p.model = object(); p.processor = object()
    p.on_disable()
    assert p.model is None and p.processor is None, "on_disable must release"
    print("[3] on_disable: model+processor released  OK")

    print("RECONFIGURE VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
