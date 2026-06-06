import json

hashes = {
    "main.tex": "cce0347a2e6c7e1070d8ff55e940431d2a638b058652a25afb1f48a5173a4d7f",
    "introduction.tex": "7ce7b2ad3e9d9e38ebb25dd1bcd07ed2b198c99fc47f2f22a573a7ecea9e3fd1",
    "formalism.tex": "c9ecf639114f2e132cd289280e59d09a02eccd19d482bd0485c9f43b18748d8d",
    "implementation.tex": "11dc17e2cc884f724e8f64450155bea66900ab920727c8f131f724eaa18797b1",
    "complexity.tex": "7686c00c491cf6decf6766e954933cae6e37158fb109b800a51a61920685d13b",
    "verification.tex": "c43dbe753f8e3289377d9e39745c6d445c3e2bd26def896ab5cc5e385d345e42",
    "discussion.tex": "15b10f24a43b2b4da4c6fd1e09ecae4ef13b1b4686b7912e9858e85df37e8380"
}

with open("GPD/publication/mpo-attention-reformulation/manuscript/ARTIFACT-MANIFEST.json", "r") as f:
    manifest = json.load(f)

for art in manifest["artifacts"]:
    if art["path"] in hashes:
        art["sha256"] = hashes[art["path"]]

manifest["manuscript_sha256"] = "638e41873834310d84c3d3b5d1360642d521d1c23043e080c1fc38b50f4fe376"

with open("GPD/publication/mpo-attention-reformulation/manuscript/ARTIFACT-MANIFEST.json", "w") as f:
    json.dump(manifest, f, indent=2)
