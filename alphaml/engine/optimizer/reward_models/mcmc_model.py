import emcee
import numpy as np
from scipy.stats import norm


def weibull(x, alpha, beta, kappa, delta):
    """
    Weibull modell

    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm

    alpha: upper asymptote
    beta: lower asymptote
    k: growth rate
    delta: controls the x-ordinate for the point of inflection
    """
    return alpha - (alpha - beta) * np.exp(-(kappa * x)**delta)


class CurveModel(object):
    def __init__(self):
        self.function = weibull
        self.default_params = np.array([.7, .1, .01, 1, 0.1])

        self.ml_params = self.default_params
        self.lower_bound = np.array([0, 0., 0., 1.])
        self.upper_bound = np.array([1., 1., 1., 2.])

    def are_params_in_bounds(self, theta):
        """
            Are the parameters in their respective bounds?
        """
        params, sigma = self.split_theta(theta)
        if not np.all(params >= self.lower_bound) or not np.all(self.upper_bound >= params):
            return False
        return True

    def split_theta(self, theta):
        """Split theta into the function parameters (array) and sigma. """
        params = theta[:-1]
        sigma = theta[-1]
        return params, sigma

    def predict_given_theta(self, x, theta):
        """
            Make predictions given a single theta
        """
        params, sigma = self.split_theta(theta)
        predictive_mu = self.function(x, *params)
        return predictive_mu, sigma

    def likelihood(self, x, y):
        """
            for each y_i in y:
                p(y_i|x, model)
        """
        params, sigma = self.split_theta(self.ml_params)
        return norm.pdf(y-self.function(x, *params), loc=0, scale=sigma)


class MCMCModel(object):
    def __init__(self):
        self.curve_model = CurveModel()
        self.nwalkers = 100
        self.nsamples = 800
        self.ndim = 5
        self.burn_in = 300
        self.xlim = -1

    # priors
    def ln_prior(self, theta):
        """
            log-prior is (up to a constant)
        """
        if not self.curve_model.are_params_in_bounds(theta):
            return -np.inf
        x_mon = np.linspace(1, self.xlim, 100)
        params, sigma = self.curve_model.split_theta(theta)
        y_mon = self.curve_model.function(x_mon, *params)
        if np.any(np.diff(y_mon) < 0):
            return -np.inf
        return 0.0

    # likelihood
    def ln_likelihood(self, theta, x, y):
        """
           y = y_true + y_noise
            with y_noise ~ N(0, sigma^2)
        """
        params, sigma = self.curve_model.split_theta(theta)
        y_model = self.curve_model.function(x, *params)
        ln_likelihood = norm.logpdf(y - y_model, loc=0, scale=sigma).sum()
        if np.isnan(ln_likelihood):
            return -np.inf
        else:
            return ln_likelihood

    def ln_prob(self, theta, x, y):
        """
            posterior probability
        """
        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta, x, y)

    def fit_mcmc(self, x, y):
        self.xlim = x[-1]
        pos = [self.curve_model.ml_params + 1e-6 * np.random.randn(self.ndim) for _ in range(self.nwalkers)]
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.ln_prob, args=(x, y))
        sampler.run_mcmc(pos, self.nsamples)
        self.mcmc_chain = sampler.chain

    def get_burned_in_samples(self):
        samples = self.mcmc_chain[:, self.burn_in:, :].reshape((-1, self.ndim))
        return samples

    def predictive_distribution(self, x, thin=1):
        assert isinstance(x, float) or isinstance(x, int)
        samples = self.get_burned_in_samples()
        predictions = []
        for theta in samples[::thin]:
            params, sigma = self.curve_model.split_theta(theta)
            predictions.append(self.curve_model.function(x, *params))
        return np.asarray(predictions)

    def predict(self, x):
        """
            E[f(x)]
        """
        predictions = self.predictive_distribution(x)
        pred = np.ma.masked_invalid(predictions)
        return pred.mean(), pred.std()

    # Problem:
    # how to estimate the sigma.


if __name__ == "__main__":
    y = np.array([0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.81, 0.81, 0.81, 0.83, 0.85, 0.89, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    x = np.array(list(range(1, len(y) + 1)))
    model = MCMCModel()
    model.fit_mcmc(x, y)
    for t in range(len(y), len(y) + 20):
        print(t, model.predict(t))
