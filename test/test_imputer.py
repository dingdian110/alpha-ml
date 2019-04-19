import pandas as pd
import numpy as np

from alphaml.engine.components.data_preprocessing.imputer import impute_df

df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'],
                  columns=["one", "two", "three"])

df["four"] = "bar"

df2 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
df2 = impute_df(df2)

print("original df:")
print(df)
print("preprocessed df:")
print(df2)
