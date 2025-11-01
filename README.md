# Pattern Recognition Assignment 3 - MLE & EM Algorithm

## Overview

Complete implementation of Maximum Likelihood Estimation (MLE) and Expectation-Maximization (EM) Algorithm for multivariate Gaussian distributions with missing data analysis.

## Single File Solution

**File:** `complete_mle_em_aic_bic_analysis.py`

This standalone Python file contains everything needed:

### Features

- ✅ **MLE Implementation** - Maximum Likelihood Estimation for Gaussian distributions
- ✅ **EM Algorithm** - Handles missing data scenarios with convergence monitoring
- ✅ **AIC/BIC Model Selection** - Compares spherical, diagonal, and full covariance models
- ✅ **Comprehensive Analysis** - Statistical validation and error analysis
- ✅ **Professional Visualizations** - Generates publication-quality plots
- ✅ **Complete Documentation** - Fully commented and documented code

### Requirements

```bash
pip install numpy scipy matplotlib
```

### Usage

Simply run the single file:

```bash
python complete_mle_em_aic_bic_analysis.py
```

### Output

The script generates:
- Complete statistical analysis output
- `complete_analysis_results.png` - Comprehensive visualization
- `aic_bic_analysis.png` - Model selection analysis

### Results Summary

- **Perfect recovery** of observed dimensions (x₁, x₂)
- **EM convergence** in ~26 iterations
- **Diagonal covariance** selected as optimal model by both AIC and BIC
- **Systematic bias** in missing dimension as expected
- **Professional visualizations** with statistical summaries

## Author

Assignment 3 - Pattern Recognition Course  
Date: November 1, 2025

---

**Note:** This is a complete, standalone implementation requiring no additional files.