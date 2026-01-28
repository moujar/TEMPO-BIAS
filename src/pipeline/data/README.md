# Methodology Dataset

This folder contains datasets for testing the political bias methodology (Fairness-AI.pdf, Fairness-AI-2.docx, 3 Methodology.docx).

## Files

| File | Description |
|------|-------------|
| **entities.csv** | 250 political figures with `entity_id`, `entity_name`, `region`, `political_spectrum`. Sampled from major geopolitical regions. |
| **templates.csv** | 60 TSC sentence templates with `template_id`, `template`, `sentiment`. Placeholder: `{entity}`. 20 positive, 20 negative, 20 neutral. |
| **methodology_dataset.csv** | Full test set: 250 × 60 = **15,000** test nodes. Columns: `sentence_id`, `template`, `entity`, `template_sentiment`. Ready for the pipeline (requires `sentence_id`, `template`, `entity`). |
| **methodology_dataset_sample.csv** | Subset for quick PoC: 50 entities × 10 templates = 500 rows. Same columns. |
| **build_methodology_dataset.py** | Script to regenerate `methodology_dataset.csv` from `entities.csv` and `templates.csv`. |

## Usage with the pipeline

Use the full or sample dataset as the `dataset.path` in your config:

```yaml
dataset:
  path: "data/methodology_dataset.csv"   # full 15,000 nodes
  # path: "data/methodology_dataset_sample.csv"  # 500 nodes for quick tests
  format: "csv"
  required_columns: ["sentence_id", "template", "entity"]
```

From the pipeline directory:

```bash
# Edit run_config.yaml to point dataset.path to data/methodology_dataset.csv or data/../data/methodology_dataset.csv
python -m tempo_bias.main --config run_config.yaml
```

## Regenerating the full dataset

```bash
cd data
python3 build_methodology_dataset.py
```

This overwrites `methodology_dataset.csv` with 15,000 rows.
