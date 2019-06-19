import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

if __name__ == "__main__":
    y = np.array(
        [0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.81, 0.81, 0.81, 0.83, 0.85, 0.89, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9])
    x = np.array(list(range(1, len(y) + 1))).reshape(-1, 1)
    model = GaussianProcessRegressor(kernel=Matern())
    model.fit(x, y)

    for t in range(len(y), len(y) + 20):
        print(t, model.predict(t, return_std=True))
