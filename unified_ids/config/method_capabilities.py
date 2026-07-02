METHOD_CAPS = {

    "mba_ocsvm_v2": {
        "binary": False,
        "prefers": "window",      # window-level scores
        "paper_dataset": "CarHacking",  # or whatever is correct, can be None
    },
    "simple_ocsvm": {
        "binary": False,
        "prefers": "window",
        "paper_dataset": "CarHacking",  # from Avatefipour
    },
    "assoc_rules": {
        "binary": False,
        "papers_dataset": "CarHacking",
    },
    "daga_ngram": {
        "binary": True,           # unseen n-gram => anomaly (no score)
        "prefers": "window",      # n-gram window
        "paper_dataset": "DAGA_STABILI2022",
    }
}
