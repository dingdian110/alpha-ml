# Supported Transformations.
## Feature Preprocessing
* imputer (missing values and NaN).
* feature encoder (text features).
* one-hot encoder (categorical features).
* label encoder (for labels).

## Feature Generation
### Scalers
* max_abs_scaler.
* min_max_scaler.
* standard_scaler.
* robust_scaler.

### Normalizers.
* normalizer. (for each sample)

### Discretizer.
* k_bins_discretizers.

### Transformers.
#### Unary transformers.
* quantile_transformer.
* polymomial_features.
* function_transformers:
    * log
    * sqrt
    * square
    * freq
    * round
    * tanh
    * sigmoid

#### N-ary transformers.
* multiplication, division, subtraction, addition.


## Feature Selection
* variance_threshold.
* generic_univariate_select (chi2, f-score, mutual-info)
* select_kbest (1x, 2x, 3x).
* select_percentile ().
* select_from_model. (lr, rf, gb)
* recursive_feature_elimination (with cv).

## Dimension Reduction
* fast_ica.
* pca (kernel pca).
* svd (truncated).
* lda.
* feature_agglomeration.

## Additional Operations
1. **balance class distribution**: balance the sample size for different labels.
2. densifier.
3. random_trees_embedding.
