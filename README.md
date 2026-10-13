# cjm-capability-voxtral-hf

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

A Mistral Voxtral speech-to-text capability for the cjm-substrate runtime that provides local transcription through Hugging Face Transformers with configurable model selection and parameter control.

## Modules

- **`cjm_capability_voxtral_hf.__init__`**
- **`cjm_capability_voxtral_hf.capability`** — Capability implementation for Mistral Voxtral transcription through Hugging Face Transformers (Option C, stage 8: pure compute).
- **`cjm_capability_voxtral_hf.degenerate`** — Degenerate-tail detection for generated transcripts (finding 84f466bb).

## API

### `cjm_capability_voxtral_hf.capability`

- `VoxtralHFCapability` _class_ — Mistral Voxtral transcription capability via Hugging Face Transformers (stage 8: pure-compute tool capability).
- `VoxtralHFCapabilityConfig` _class_ — Configuration for Voxtral HF transcription capability.

### `cjm_capability_voxtral_hf.degenerate`

- `find_degenerate_onset` _function_ — Locate the earliest position where a 1..max_ngram-word phrase repeats
- `truncate_degenerate_tail` _function_ — Cut a runaway repetition tail, keeping the prefix VERBATIM plus one

## Dependencies

**Depends on:** `accelerate`, `cjm-capability-primitives`, `cjm-substrate`, `cjm-substrate-hf-utils`, `cjm-substrate-torch-utils`, `fastcore`, `librosa`, `mistral-common`, `numpy`, `soundfile`, `torch`, `transformers`
**Used by:** `cjm-transcription-core`
