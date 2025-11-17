# Harmonize: Data Harmonization System using LLM

A comprehensive system for data harmonization using LLM. Benchmarked to multiple matching approaches including COMA, LLM-based matching, embeddings, clustering, and ensemble methods.

## Table of Contents

- [Project Overview](#project-overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Approaches](#approaches)
- [Known Issues](#known-issues)
- [Contributing](#contributing)

---

## Project Overview

Harmonize provides a unified framework for schema matching across multiple datasets (Valentine, synthetic data, etc.). It supports:

- **Schema Inference**: Automatically infer schemas from CSV files
- **Multiple Matching Approaches**: COMA, LLM (GPT-5), embeddings, clustering, edit distance, and ensemble methods
- **Confidence Calibration**: Calibrate prediction confidence scores
- **Evaluation**: Compute accuracy metrics and statistical analysis

---

## System Requirements

### Hardware

- **RAM**: Minimum 6–8 GB (COMA matcher requires significant memory)
- **Disk Space**: 10+ GB for datasets and cached results
- **CPU**: Multi-core processor recommended for parallel processing

### Software

- **Python**: 3.10 or higher
- **Java**: JRE/JDK 11+ (required for COMA matcher)
- **Git**: For version control and Git LFS for large files

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Fabu1da/harmonize.git
cd harmonize
```

### 2. Install Java (if not already installed)

On macOS:

```bash
brew install openjdk
# Add to PATH (follow brew instructions, typically):
echo 'export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
java -version
```

On Linux (Ubuntu/Debian):

```bash
sudo apt-get install openjdk-11-jre
java -version
```

On Windows:
Download and install from [Oracle Java](https://www.oracle.com/java/technologies/javase-downloads.html) or use Chocolatey:

```bash
choco install openjdk
```

### 3. Install Python Dependencies

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### 4. Install Git LFS (for large files)

```bash
brew install git-lfs  # macOS
# or
sudo apt-get install git-lfs  # Linux
# or download from https://git-lfs.github.com

git lfs install
git lfs pull  # Download large files tracked by LFS
```

### 5. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env  # if available
# or create manually:
echo "OPENAI_API_KEY=your_key_here" > .env
```

Add your API keys (for LLM-based approaches):

```
OPENAI_API_KEY=sk-...
```

---

## Configuration

### Main Configuration File: `config.py`

The `config.py` file defines all matching approaches and their parameters:

```python
APPROACHES: list[tuple[BaseApproach, str]] = [
    (RandomApproach(), "Random"),
    (NullApproach(), "Null"),
    (IdentityApproach(), "Identity"),
    (EditDistanceApproach(threshold=0.1), "Edit Distance"),
    (ComaApproach(use_instances=False), "COMA (without instances)"),
    (ComaApproach(use_instances=True), "COMA (with instances)"),
    (LLMApproach(model="gpt-5-mini"), "LLM (GPT-5 mini)"),
    (ClusteringApproach(model="text-embedding-3-large"), "Clustering (Text Embedding Large)"),
    # ... more approaches
]
```

### COMA Memory Configuration

If you encounter Java memory errors, edit `run_coma_example.py`:

```python
# Line ~45
matcher = Coma(java_xmx="8192m", use_instances=use_instances)
# Change to match your available RAM (e.g., 4096m for 4 GB):
matcher = Coma(java_xmx="4096m", use_instances=use_instances)
```

---

## Usage

### Quick Start

#### 1. Run Complete Pipeline (Predict → Train → Calibrate → Evaluate)

```bash
python main.py --ptce
```

#### 2. Run Only Prediction

```bash
python main.py --predict
```

Options:

- `--no-cache`: Disable schema caching; re-infer schemas each run
- `--seed 42`: Set random seed for reproducibility

#### 3. Run Only Training (Confidence Calibrators)

```bash
python main.py --train
```

#### 4. Run Only Calibration

```bash
python main.py --calibrate
```

#### 5. Run Only Evaluation

```bash
python main.py --evaluate
```

#### 6. Make reports

```bash
python main_report.py
```

### Run Missing COMA (with instances) Predictions

If a previous run was interrupted and you have 36 missing predictions:

```bash
python scripts/run_missing_coma_with_instances.py
```

Options:

- `--dry-run`: List missing expected files without running predictions
- `--limit=N`: Limit to N files (e.g., `--limit=5`)

### Generate Synthetic Data

```bash
python main.py --generate-synthetic
```

### Run Specific Tests

```bash
# Run McNemar's test on predictions:
python mcnemar.py

# Generate performance reports:
python reports/report.py
```

---

## Project Structure

```
harmonize/
├── main.py                           # Main entry point
├── config.py                         # Approach configuration
├── requirements.txt                  # Python dependencies
├── run_coma_example.py              # COMA wrapper (Valentine)
├── schema_inference.py              # Schema inference logic
├── embedding_utils.py               # Embedding utilities
├── gpt_utils.py                     # GPT/LLM utilities
├── json_schema.py                   # Schema data model
│
├── approaches/                       # Matching approach implementations
│   ├── base.py                      # Base approach class
│   ├── coma.py                      # COMA approach
│   ├── llm.py                       # LLM approach
│   ├── embedding.py                 # Embedding-based approach
│   ├── cluster.py                   # Clustering approach
│   ├── ensemble.py                  # Ensemble approaches
│   └── ...                          # Other approaches
│
├── assets/                           # Data and results
│   ├── expected/                    # Ground truth mappings
│   │   ├── valentine/               # Valentine benchmark datasets
│   │   └── synthetic/               # Synthetic test data
│   ├── source/                      # Source table CSVs
│   ├── target/                      # Target table CSVs
│   └── predicted/                   # Predictions by approach
│       ├── COMA (with instances)/
│       ├── LLM (GPT-5 mini)/
│       ├── Embedding.../
│       └── ...
│
├── models/                           # Trained models
│   └── confidence_calibrators/      # Calibrator models
│
├── reports/                          # Analysis and reporting
│   ├── report.py                    # Main report generation
│   ├── agreement.py                 # Inter-approach agreement analysis
│   ├── statistical_analysis.py      # Statistical tests
│   └── output/                      # Generated reports (LaTeX, tables)
│
├── scripts/                          # Utility scripts
│   └── run_missing_coma_with_instances.py  # Resume interrupted runs
│
├── core/                             # Core utilities
│   ├── compare_mappings.py          # Mapping comparison logic
│   └── utils/
│       ├── save_calibrator.py       # Model persistence
│       └── synthetic.py             # Synthetic data generation
│
└── venv/                             # Python virtual environment (gitignored)
```

---

## Approaches

### Available Matching Approaches

| Approach          | Type         | Configuration                    | Notes                                                  |
| ----------------- | ------------ | -------------------------------- | ------------------------------------------------------ |
| **Random**        | Baseline     | -                                | Random column matching (for comparison)                |
| **Null**          | Baseline     | -                                | No matches (for comparison)                            |
| **Identity**      | Baseline     | -                                | Match columns with identical names                     |
| **Edit Distance** | String-based | `threshold=0.1`                  | String similarity matching                             |
| **COMA**          | Hybrid       | `use_instances=True/False`       | Instance + schema-based matching (requires Java)       |
| **LLM**           | Semantic     | `model="gpt-5-mini"`             | GPT-based matching (requires API key)                  |
| **Clustering**    | Embedding    | `model="text-embedding-3-large"` | Clustering-based similarity                            |
| **Embedding**     | Embedding    | `method="greedy"/"hungarian"`    | Embedding similarity with greedy or optimal assignment |
| **Ensemble**      | Ensemble     | `approaches=[...]`               | Majority vote or weighted ensemble                     |

### COMA Approach Details

COMA (Cognitive Modeling of Analogy) uses both schema and instance data:

- **Without instances** (`use_instances=False`): Schema only, faster, lower memory
- **With instances** (`use_instances=True`): Uses actual table data, better accuracy, higher memory/time

**Memory requirements**: ~4–8 GB JVM heap depending on table size. Adjust `java_xmx` in `run_coma_example.py` if you encounter `JavaException`.

---

## Known Issues

### 1. Java Memory Errors

**Error**: `JavaException('Either Java (JRE) is not installed or Java does not have enough memory...')`

**Solution**:

- Check Java is installed: `java -version`
- Reduce JVM heap in `run_coma_example.py`: change `java_xmx="8192m"` to `java_xmx="4096m"`
- Use sampling for large tables (see scripts)

### 2. CSV Parsing Issues

**Issue**: Some datasets have many columns with embedded commas or inconsistent quoting.

**Solution**:

- The system uses pandas with `dtype=str` and `low_memory=False` for robust parsing
- If issues persist, manually check CSV format and re-save with proper quoting

### 3. Missing Predictions After Interruption

**Issue**: Process interrupted; some expected files don't have predictions yet.

**Solution**:

```bash
python scripts/run_missing_coma_with_instances.py
```

This script skips already-predicted files and runs only missing ones.

### 4. Large File Upload (Git)

**Issue**: Cache files exceed GitHub's 100 MB limit.

**Solution**: Use Git LFS (already configured):

```bash
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Track large files with LFS"
git push
```

---

## Troubleshooting

### Virtual Environment Issues

```bash
# Recreate venv if needed:
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Missing Dependencies

```bash
# Reinstall all dependencies:
pip install --upgrade -r requirements.txt
```

### Schema Caching

Clear cached schemas if they become stale:

```bash
rm -rf assets/**/*.csv.json
python main.py --predict --no-cache
```

---

## Performance Tuning

### For Large Datasets

1. **Increase Java heap** (if you have RAM):

   ```python
   matcher = Coma(java_xmx="16384m", ...)  # 16 GB
   ```

2. **Run in parallel** (with caution on concurrent COMA):
   - Most approaches support async; ensemble may benefit
   - Sequential is safer for COMA to avoid memory conflicts

### For Memory-Constrained Systems

1. Lower Java heap: `java_xmx="2048m"` (2 GB)
2. Use instance-free COMA: `use_instances=False`
3. Skip embedding approaches or use smaller models
4. Run one approach at a time, clear cache between runs

---

## Output & Results

### Prediction Files

Predictions are saved in:

```
assets/predicted/{approach_name}/{dataset_name}.json
```

Format:

```json
{
  "target_column_1": ["source_column_1", 0.95, "Match reason"],
  "target_column_2": [null, 0.0, "No match found"],
  ...
}
```

### Reports & Evaluation

Generated reports are saved in:

```
assets/reports/
output/
```

Key files:

- `performance_table.tex` — Main performance comparison
- `statistical_analysis.tex` — McNemar's test results
- `agreement_table.tex` — Inter-approach agreement matrix
- `confidence_calibration_plot.tex` — Calibration curves

---

## Contributing

To add a new matching approach:

1. Create a new file in `approaches/`:

   ```python
   # approaches/my_approach.py
   from .base import BaseApproach

   class MyApproach(BaseApproach):
       async def predict(self, source_schema, target_schema, **kwargs):
           # Your implementation
           return predictions
   ```

2. Register in `config.py`:

   ```python
   (MyApproach(), "My Approach Name"),
   ```

3. Test and run:
   ```bash
   python main.py --predict
   ```

---

## References

- **Valentine**: https://zenodo.org/records/5084605
- **COMA**: Schema matching algorithm
- **GPT Models**: OpenAI API
- **Embeddings**: OpenAI text-embedding-3-large

---

**Last Updated**: November 2025
