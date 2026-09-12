"""Degenerate-tail detection for generated transcripts (finding 84f466bb).

Voxtral-mini locks into a repetition loop on some lecture chunks — coherent
text for a while, then one short phrase repeated to the token cap ('some'
× 24790; 'The precomposition.' × 2402). Sampling (do_sample / temperature)
changes the phrase, not the degeneration, and a runaway text downstream is
what spiked forced alignment to 15GB (the admission self-deadlock 3993a755).
The text BEFORE the loop is usually better than whisper's, so the guard
TRUNCATES at the loop onset (keeping one instance of the phrase) and marks
the result — never discards the good prefix, never raises.

Pure functions: no torch, no model — unit-testable in any env.
"""

import re
from typing import Any, Dict, List, Optional


def find_degenerate_onset(
    words: List[str],       # Whitespace-split tokens of the generated text
    min_repeats: int = 8,   # Consecutive repeats of one phrase that count as a loop
    max_ngram: int = 6,     # Longest phrase length (in words) to test
) -> Optional[Dict[str, Any]]:  # {onset, ngram, phrase, repeats} at the EARLIEST loop, or None
    """Locate the earliest position where a 1..max_ngram-word phrase repeats
    itself at least `min_repeats` times back to back.

    Scans onsets in order and returns at the first hit, so the cost before a
    loop is O(words × max_ngram) with tiny inner counts; at the onset itself
    one count runs to the end of the loop. Natural speech repeats a word a few
    times ('no, no, no'); eight identical consecutive repeats of the same
    phrase is the runaway signature, not speech.
    """
    n_words = len(words)
    if min_repeats < 2 or n_words < min_repeats:
        return None
    for i in range(n_words):
        for n in range(1, max_ngram + 1):
            if i + n * min_repeats > n_words:
                break
            phrase = words[i:i + n]
            repeats = 1
            while words[i + repeats * n:i + (repeats + 1) * n] == phrase:
                repeats += 1
            if repeats >= min_repeats:
                return {"onset": i, "ngram": n, "phrase": " ".join(phrase),
                        "repeats": repeats}
    return None


def truncate_degenerate_tail(
    text: str,              # The generated transcript text
    min_repeats: int = 8,   # Consecutive repeats that count as a loop (0 disables)
    max_ngram: int = 6,     # Longest phrase length tested
) -> Dict[str, Any]:  # {"text": kept text, "degenerate_tail": None | {...}}
    """Cut a runaway repetition tail, keeping the prefix VERBATIM plus one
    instance of the looping phrase.

    Returns the (possibly unchanged) text and a marker describing the cut —
    onset word index, the phrase, how many times it repeated, and the word
    counts before/after — so the transcript's metadata carries the evidence.
    `min_repeats <= 0` disables the detector (text passes through unchanged).
    """
    if min_repeats <= 0:
        return {"text": text, "degenerate_tail": None}
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    words = [text[a:b] for a, b in spans]
    hit = find_degenerate_onset(words, min_repeats=min_repeats, max_ngram=max_ngram)
    if hit is None:
        return {"text": text, "degenerate_tail": None}
    keep_words = hit["onset"] + hit["ngram"]  # prefix + ONE instance of the phrase
    cut_at = spans[keep_words - 1][1]
    return {
        "text": text[:cut_at],
        "degenerate_tail": {
            "onset_word": hit["onset"],
            "phrase": hit["phrase"],
            "repeats": hit["repeats"],
            "words_before": len(words),
            "words_after": keep_words,
        },
    }
