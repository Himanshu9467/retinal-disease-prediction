from pathlib import Path

for split in ["train","val","test"]:
    normal = len(list(Path(f"research_binary_7000/{split}/normal").glob("*.*")))
    disease = len(list(Path(f"research_binary_7000/{split}/disease").glob("*.*")))

    print(split, normal, disease)