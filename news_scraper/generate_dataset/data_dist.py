import json
from collections import Counter
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "../data/train_data/combined/train.jsonl"
counts = Counter()

with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        news_section = rec["messages"][1]["content"].split("News:\n", 1)[-1]
        n = len([l for l in news_section.splitlines() if l.strip()])
        counts[n] += 1

total = sum(counts.values())
print(f"{path}: {total} rows\n")
for k in sorted(counts):
    print(f"  {k} news items: {counts[k]:>5} rows ({counts[k]/total*100:.1f}%)")

mean_n = sum(k * v for k, v in counts.items()) / total
print(f"\nMean news items/row: {mean_n:.2f}")