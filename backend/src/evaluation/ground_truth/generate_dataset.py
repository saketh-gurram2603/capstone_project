"""
Ground truth dataset generator.

Reads the actual incidents from the ingested data and creates a structured
evaluation dataset with query/relevant_ids/expected_answer triplets.

Usage:
    python -m src.evaluation.ground_truth.generate_dataset \
        --xlsx data/incidents.xlsx \
        --output src/evaluation/ground_truth/dataset.json \
        --n 30

The generated dataset is committed to the repo so the eval runner can load it
without requiring the original XLSX at test time.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def generate_from_xlsx(xlsx_path: str, n: int = 30, seed: int = 42) -> list[dict]:
    """
    Build evaluation test cases from the actual incident XLSX.

    Strategy:
      - For each test case, pick a random incident as the "anchor"
      - Use its category + description as the query
      - Mark same-category incidents as relevant (up to 3)
      - Use the anchor's solution as the expected answer
    """
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas required: pip install pandas openpyxl")

    df = pd.read_excel(xlsx_path)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    expected_cols = ["Incident ID", "Category", "Description", "Solution"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {xlsx_path}. Found: {list(df.columns)}")

    df = df.dropna(subset=["Incident ID", "Description", "Solution"])
    df = df.reset_index(drop=True)

    random.seed(seed)
    anchors = random.sample(list(df.index), min(n, len(df)))

    dataset = []
    for i, idx in enumerate(anchors):
        row = df.iloc[idx]
        incident_id = str(row["Incident ID"]).strip()
        category = str(row.get("Category", "General")).strip()
        description = str(row["Description"]).strip()
        solution = str(row["Solution"]).strip()

        # Relevant: same category incidents (excluding self)
        same_cat = df[
            (df["Category"].astype(str).str.strip() == category) &
            (df.index != idx)
        ]
        relevant_ids = [str(r["Incident ID"]).strip() for _, r in same_cat.head(3).iterrows()]
        # Always include the anchor itself as relevant
        relevant_ids = [incident_id] + relevant_ids

        dataset.append({
            "id": f"GT-{i + 1:03d}",
            "query": description[:300],
            "category": category,
            "relevant_incident_ids": relevant_ids[:4],
            "expected_answer": solution[:500],
        })

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation ground truth dataset")
    parser.add_argument("--xlsx", default="data/incidents.xlsx", help="Path to incidents XLSX")
    parser.add_argument(
        "--output",
        default="src/evaluation/ground_truth/dataset.json",
        help="Output JSON path",
    )
    parser.add_argument("--n", type=int, default=30, help="Number of test cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Generating {args.n} test cases from {args.xlsx} ...")
    dataset = generate_from_xlsx(args.xlsx, n=args.n, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} test cases to {output_path}")

    # Print category distribution
    from collections import Counter
    cats = Counter(tc["category"] for tc in dataset)
    print("Category distribution:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
