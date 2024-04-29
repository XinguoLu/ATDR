# ATDR

## paper " Drug Repositioning Method with Attention Transfer Across Networks on Deep Feature Representation for Disease Treatment"

### 'dataset' directory

Contain the gold standard drug-disease set and four drug-related networks.

### 'preprocessing' directory

Contain the preprocessing code to generate PPMI matrix.

### 'PPMI_CSV' directory

Contain the PPMI matrices of 4 drug-related networks.

### 'AT' directory

attention transfer basic model

### Tutorial

1. Create two directories "test_models" and "results" in the project.
2. To get drug features learned by MDA, run

  - python getFeatures.py

3. To predict drug-disease associations through attention transfer, run

  - python prediction.py

### Requirements

ATDR is tested to work under Python 3.9. The required packages can be found in requirements.txt.