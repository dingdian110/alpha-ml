# Feature Engineering Operators
All the operators are under directory `alphaml/engine/components/pipeline`  
Each operator has a method *operate(dm_list:List,  phase:{'train','test'})*
1. Parameter *dm_list*: A list of inputs.  
For InputerOperator, it contains a single DataFrame.  
For Data Preprocessing Operators and Feature Generation Operators, it contains a single DataManager.  
For Feature Selection Operators, it contains at least one DataManager.
2. The parameter *phase*: A string from {'train','test'}.  
If set to 'train', the operator will fit the inner operator(or estimator) and transform the inputs.
If set to 'test', the operator will transfrom the inputs according to the fitted inner operator.
3. The method *operate* returns a DataManager

## Data Preprocessing Operators
- **InputerOperator**: *params* None
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
    - *params* stands for degrees

- **PCAOperator** *params* int
    - *params* stands for n_components (the number of features needed)
    
- **AutoCrossOperator:** *params* int ***Not yet finished***



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