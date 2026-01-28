#!/usr/bin/env python3
"""
Build the full methodology dataset: 250 entities × 60 templates = 15,000 test nodes.
Output CSV columns: sentence_id, template, entity, template_sentiment
"""
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

def main():
    with open(DATA_DIR / "entities.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        entity_names = [row["entity_name"] for row in reader]

    with open(DATA_DIR / "templates.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        template_rows = list(reader)

    out_path = DATA_DIR / "methodology_dataset.csv"
    count = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence_id", "template", "entity", "template_sentiment"])
        writer.writeheader()
        sentence_id = 1
        for tr in template_rows:
            template = tr["template"]
            sentiment = tr["sentiment"]
            for entity in entity_names:
                writer.writerow({
                    "sentence_id": sentence_id,
                    "template": template,
                    "entity": entity,
                    "template_sentiment": sentiment,
                })
                sentence_id += 1
                count += 1

    print(f"Wrote {count} rows to {out_path}")
    print(f"  Entities: {len(entity_names)}")
    print(f"  Templates: {len(template_rows)}")
    print(f"  Test nodes: {count}")

if __name__ == "__main__":
    main()
