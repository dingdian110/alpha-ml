# Feature Engineering Hyperparameters
The operators are under `alphaml/engine/components/pipeline`
## Data Preprocessing Operators
- **LabelEncoderOperator:** *params* None
- **FeatureEncoderOperator:** *params* {0,1}  
    - 0 for One-hot Encoder
    - 1 for Ordinal Encoder
- **ScalerOperator:** *params* {0,1,2}
    - 0 for Standard Scaler
    - 1 for Minmax Scaler
    - 2 for Maxabs Scaler
- **NormalizerOperator:** *params* {0,1}  
    - 0 for L2 norm
    - 1 for L1 norm
    
## Feature Generation Operators
- **PolynomialFeaturesOperator:** *params* int
    - *params* stand for degrees
    
- **AutoCrossOperator:** *params* int
    - *params* stand for degrees

## Feature Selection Operators
- **NaiveSelectorOperator:** *params* list [int,  {0,1,2,3,4}]
    - The first element stands for k in k-best
    - The second element stands for metric
        - 0 for chi2 (cls/reg, non-negative features)
        - 1 for f_classif (cls)
        - 2 for mutual_info_classif (cls)
        - 3 for f_regression (rgs)
        - 4 for mutual_info_regression (rgs)
- **MLSelectorOperator:** *params* list [int,  {0,1}, {0,1}]
    - The first element stands for k in k-best
    - The second element stands for task type
        - 0 for classification
        - 1 for regression
    - The third element stands for ML model
        - 0 for RandomForest
        - 1 for Lasso (or LogisticRegression with L1 penalty)