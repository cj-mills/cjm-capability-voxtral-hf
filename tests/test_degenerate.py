"""Tests for cjm_capability_voxtral_hf.degenerate — the runaway-repetition guard
(finding 84f466bb). Pure functions: no torch, no model."""
from cjm_capability_voxtral_hf.degenerate import (find_degenerate_onset,
                                                  truncate_degenerate_tail)


def test_single_word_runaway_is_cut_at_onset_keeping_one_instance():
    # The live exhibit: 169 coherent words, then 'some' × 24790 to the token cap.
    prefix = "Eight elements instead of one element. And the dequantization function"
    text = prefix + " some" * 300
    out = truncate_degenerate_tail(text, min_repeats=8)
    assert out["text"] == prefix + " some"
    tail = out["degenerate_tail"]
    assert tail["phrase"] == "some" and tail["repeats"] == 300
    assert tail["onset_word"] == len(prefix.split())
    assert (tail["words_before"], tail["words_after"]) == (len(prefix.split()) + 300,
                                                            len(prefix.split()) + 1)


def test_multi_word_phrase_loop_is_detected():
    # Under sampling the phrase changes, not the loop: 'The precomposition.' × 2402.
    prefix = "So the trick is"
    text = prefix + " The precomposition." * 40
    out = truncate_degenerate_tail(text, min_repeats=8)
    assert out["text"] == prefix + " The precomposition."
    assert out["degenerate_tail"]["phrase"] == "The precomposition."
    assert out["degenerate_tail"]["repeats"] == 40


def test_natural_repetition_below_threshold_passes_through_verbatim():
    # A speaker saying 'no' four times is speech (the FindBar exhibit 9cd95a20),
    # not a loop; whitespace and punctuation in the prefix stay untouched.
    text = "No, no, no, no — that's not it.\nIt was the\tother one."
    out = truncate_degenerate_tail(text, min_repeats=8)
    assert out["text"] == text and out["degenerate_tail"] is None


def test_threshold_zero_disables_the_guard():
    text = "yes " * 50
    out = truncate_degenerate_tail(text, min_repeats=0)
    assert out["text"] == text and out["degenerate_tail"] is None


def test_earliest_onset_wins_and_prefix_is_verbatim():
    text = "a b c " + "x y " * 10 + "tail words here " + "z " * 20
    hit = find_degenerate_onset(text.split(), min_repeats=8)
    assert hit == {"onset": 3, "ngram": 2, "phrase": "x y", "repeats": 10}
    out = truncate_degenerate_tail(text, min_repeats=8)
    assert out["text"] == "a b c x y"
