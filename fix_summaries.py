import os
import re

for root, dirs, files in os.walk("GPD/phases"):
    for file in files:
        if file.endswith("SUMMARY.md"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                content = f.read()
            
            # Add missing fields if not present
            if "depth:" not in content:
                content = content.replace("status: complete\n", "status: complete\ndepth: full\n", 1)
                content = content.replace("status: completed\n", "status: completed\ndepth: full\n", 1)
            
            if "provides:" not in content:
                content = content.replace("depth: full\n", "depth: full\nprovides: [phase-work]\n", 1)
            
            if "completed: true" not in content:
                content = content.replace("provides: [phase-work]\n", "provides: [phase-work]\ncompleted: true\n", 1)

            with open(path, "w") as f:
                f.write(content)
