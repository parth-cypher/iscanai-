import splitfolders

splitfolders.ratio(
    "archive (1)/dataset",          # your current folder
    output="output",    # new folder
    seed=42,
    ratio=(0.8, 0.1, 0.1)
)