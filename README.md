# CULT - AI-Powered Candidate Ranking System

## Overview
This repository contains the solution for the Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge.
It strictly adheres to the < 5 minute, CPU-only, and no-network constraints for the online ranking phase.

## Setup
1. Create a virtual environment (optional).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow

### 1. Offline Pre-computation Phase
Run the offline processor to extract features, compute embeddings, detect honeypots, and apply hard filters.
This step processes the raw `candidates.jsonl` and generates a fast, compact cache file (`features_cache.parquet`).
*Note: This phase operates outside the 5-minute timed window and requires network access to download the `all-MiniLM-L6-v2` embedding model initially.*

```bash
python offline_processor.py
```

### 2. Timed Ranking Phase
Run the fast ranker to compute final scores based on the offline-extracted features, format the factual reasoning column, and generate the final submission CSV.
This step guarantees completion well under the 5-minute limit and operates strictly CPU-only with no network access.

```bash
python ranker.py
```

### 3. Validation
Validate the submission using the provided script:
```bash
python validate_submission.py CULT.csv
```
