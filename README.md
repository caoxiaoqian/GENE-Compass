# GENE-Compass: A Universal Computational Framework for GRN Inference in Plants

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

## Introduction

GENE-Compass is an open-source computational framework for inferring gene regulatory networks (GRNs) in plants, with a focus on high precision and efficiency using large-scale, noisy transcriptome data. Inspired by the research in [GENE-Compass: A Universal Computational Framework for GRNs Inference in Plants Using Extreme Correlation Filtering and Machine Learning](https://arxiv.org/abs/XXXX.XXXXX) (manuscript under review, 2025), this toolkit implements a three-tier pipeline:

1. **Extreme Correlation Filtering**: Leverages statistical power from RNA-seq data to select high signal-to-noise co-expression pairs (e.g., top 5% Pearson correlation tails).
2. **Biological Prior Filtering**: Integrates biophysical priors like position weight matrices (PWMs) via FIMO motif scanning for promoter regions.
3. **Efficient Statistical Inference (PORTIA)**: Applies causal inference with synergistic gains from priors, achieving >40x speedup over competitors.

The framework excels in both model plants (e.g., *Arabidopsis thaliana*, 66.7% Y1H validation) and non-model plants (e.g., rose, 42.9% Y1H validation) via orthology-based PWM transfer across 157 species.

This repository provides Python tools for promoter extraction, motif scanning, correlation analysis, and GRN inference, enabling reproducible GRN reconstruction.

## Features

- **Cross-Species Compatibility**: PWM knowledge transfer for non-model plants using orthology (e.g., MMseqs2 alignment).
- **High Efficiency**: Parallel processing with Joblib, multiprocessing, and optimized algorithms (e.g., vectorized correlations).
- **Validation-Ready Outputs**: Ranked predictions with enrichment factors (EF), AUC-EF curves, and Y1H-compatible formats.
- **Modular Design**: Interchangeable GRN methods (Random Forest, XGBoost, SVR, PORTIA) with normalization (z-score, min-max).
- **Visualization**: Built-in plotting for EF curves, ranking enrichments, and experimental diagrams (via Matplotlib).

Key benchmarks:
- *Arabidopsis* AtWRKY33: 66.7% Y1H success (6/9 validated interactions).
- Rose RhWRKY23: 42.9% Y1H success (6/14 validated), robust across top-600 rankings.

## Installation

### Prerequisites
- Python 3.8+
- Biopython, NumPy, SciPy, Pandas, Matplotlib, Scikit-learn, XGBoost, Joblib
- MEME Suite (for FIMO/PWM handling): Install via [official guide](https://meme-suite.org/docs/installation/).
- GimmeMotifs (optional, for alternative scanning): `pip install gimmemotifs`
- tqdm (for progress bars): `pip install tqdm`

### Quick Setup
```bash
git clone https://github.com/your-repo/gene-compass.git  # Replace with actual repo
cd gene-compass
pip install -r requirements.txt  # Create this file with deps above
# For MEME: Download and add to PATH (e.g., export PATH=$PATH:/path/to/meme/bin)
```

Example `requirements.txt`:
```
biopython>=1.79
numpy>=1.21
scipy>=1.7
pandas>=1.3
matplotlib>=3.4
scikit-learn>=1.0
xgboost>=1.5
joblib>=1.1
tqdm>=4.62
gimmemotifs>=0.15  # Optional
```

## Usage

### 1. Extract Promoters
Use `exact_promoter.py` to extract 2kb upstream promoter sequences from GFF3/genome FASTA.

```bash
python exact_promoter.py \
  --gff ./odata/Rosa_chinensis/Rosa_chinensis.RchiOBHm-V2.60.gff3 \
  --genome ./odata/Rosa_chinensis/Rosa_chinensis.RchiOBHm-V2.dna_sm.toplevel.fa \
  -o Rosa_chinensis_promoter.fasta \
  -u 2000  # Upstream bases
```

Output: FASTA with uppercase upstream/lowercase downstream sequences, annotated with TSS anchors.

### 2. Motif Scanning (FIMO or GimmeMotifs)
Scan promoters for TF motifs using `fimo_scan.py` (MEME-based) or `mscan_utils.py` (GimmeMotifs).

**FIMO Example** (parallel, multi-core):
```bash
python fimo_scan.py \
  -md /path/to/motifs/  # Recursive *.meme dir
  -f Rosa_chinensis_promoter.fasta \
  -t 12  # Score threshold
  -rc  # Scan reverse complement
  -j 8  # 8 jobs
  -o fimo_results.csv
```

**GimmeMotifs Example**:
```bash
python mscan_utils.py \
  -m /path/to/motifs/ \
  -g /path/to/genome.fa \
  -f Rosa_chinensis_promoter.fasta \
  --fpr 0.01 \
  --score-threshold 10 \
  -o gimmemotifs_results.csv
```

Output: CSV with seq_name, motif, score, position, strand, crop_seq, p/q-values.

### 3. Extreme Correlation Filtering
Use `exppc.py` for vectorized Pearson correlations and percentile analysis.

```python
import numpy as np
from exppc import vectorized_correlation, percentile_analysis

# Example: target_vector (samples x 1), matrix (samples x genes)
corr, pvals = vectorized_correlation(target_vector, expression_matrix)
analysis = percentile_analysis(corr, alpha=0.025)  # Top/bottom 5%
extreme_pairs = np.where((corr <= analysis['percentiles']['lower']) | (corr >= analysis['percentiles']['upper']))[0]
```

### 4. GRN Inference
Core pipeline in `grn.py`: Filter co-expressions, apply motif priors, infer with PORTIA/RF/XGBoost.

```python
import pandas as pd
from grn import parallel_grn, Meta_Grn_Optimized_v3
from io_utils import for_datafream  # For adj matrix to ranked DF

# Load data: expression (samples x genes), tg2tfs (dict: target -> TF list)
expression_data = pd.read_csv('expression.csv', index_col=0)
tg2tfs = {'RhWRKY23': ['TF1', 'TF2', ...]}  # From correlation filter

# Parallel inference (non-PORTIA)
results = parallel_grn(tg2tfs, expression_data, grn_methods=['xgboost', 'rf'], n_jobs=-1, normalization_method='zscore')

# Single-target with PORTIA
adj_matrix = Meta_Grn_Optimized_v3(expression_data, tf_names=tg2tfs['target'], target_gene_name='target', grn_methods=['portia']).grn_results['portia']

# Ranked output
ranked_df = for_datafream(adj_matrix, 'TF_name')
ranked_df.to_csv('grn_predictions.csv')
```

For full pipeline integration, chain with `utils.py` for FIMO filtering:
```python
from utils import fimo_filter
scanned_genes = fimo_filter(tf='RhWRKY23', motif_infos=motif_df, pcc_genes=extreme_pairs, motif_scan_res=fimo_df)
```

## Examples

See `./examples/` (create folder with Jupyter notebooks):
- `rose_grn_inference.ipynb`: End-to-end rose RhWRKY23 prediction.
- `arabidopsis_benchmark.ipynb`: AtWRKY33 validation plotting.

## Contributors

Thank you to all contributors for making this project even better!

| Contributor              | GitHub                  |
|--------------------------|-------------------------|
| Xiaoqian Cao             | [@caoxiaoyan](https://github.com/caoxiaoyan) |
| Yao Huang                | [@yaohuang](https://github.com/yaohuang110)     |
| Jicheng Tang             | [@Jicheng-Tang](https://github.com/Jicheng-Tang)  |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GENE-Compass, please cite:

> Cao, X., Huang, Y., Watcharatpong, P., Han, R., & Zhang, Z. (2025). GENE-Compass: A Universal Computational Framework for GRNs Inference in Plants Using Extreme Correlation Filtering and Machine Learning. *bioRxiv*. [DOI: XXXX.XXXXX](https://doi.org/XXXX.XXXXX)

For code contributions or issues, open a GitHub issue. Contributions welcome! 🚀
