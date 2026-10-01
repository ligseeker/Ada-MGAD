from scripts.p6.analyze_c0_causal_development import paired_table


def test_paired_recall_keeps_all_misses_and_original_denominator():
    report = paired_table(["both", "lost", "gain", "miss"],
                          {"both", "lost"}, {"both", "gain"})
    assert report["n"] == 4
    assert report["both_hit"] == report["both_miss"] == 1
    assert report["candidate_only"] == report["baseline_only"] == 1
    assert report["delta_recall"] == 0
    subset = paired_table(["gain", "miss"], {"both", "lost"}, {"both", "gain"})
    assert subset["n"] == 2 and subset["delta_recall"] == .5
