from typing import Union
import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import scipy.stats as st
from .exceptions import NotEnoughParticles
from .base import Transition
from .util import smart_cov


def scott_rule_of_thumb(n_samples, dimension):
    """
    Scott's rule of thumb.

    .. math::

       \\left ( \\frac{1}{n} \\right ) ^{\\frac{1}{d+4}}

    (see also scipy.stats.kde.gaussian_kde.scotts_factor)
    """
    return n_samples ** (-1. / (dimension + 4))


def silverman_rule_of_thumb(n_samples, dimension):
    """
    Silverman's rule of thumb.

    .. math::

       \\left ( \\frac{4}{n (d+2)} \\right ) ^ {\\frac{1}{d + 4}}

    (see also scipy.stats.kde.gaussian_kde.silverman_factor)
    """
    return (4 / n_samples / (dimension + 2)) ** (1 / (dimension + 4))




class MultivariateNormalTransition(Transition):
    """
    Transition via a multivariate Gaussian KDE estimate.

    Parameters
    ----------

    scaling: float
        Scaling is a factor which additionally multiplies the
        covariance with. Since Silverman and Scott usually have too large
        bandwidths, it should make most sense to have 0 < scaling <= 1

    bandwidth_selector: optional
        Defaults to `silverman_rule_of_thumb`.
        The bandwidth selector is a function of the form
        f(n_samples: float, dimension: int),
        where n_samples denotes the (effective) samples size (and is therefore)
        a float and dimension is the parameter dimension.

    """
    def __init__(self, scaling=1, bandwidth_selector=silverman_rule_of_thumb,
                 ):
        self.scaling = scaling
        self.bandwidth_selector = bandwidth_selector

    def fit_cov(self, x, w):

        sample_cov = smart_cov(x, w)
        dim = sample_cov.shape[0]
        eff_sample_size = 1 / (w**2).sum()
        bw_factor = self.bandwidth_selector(eff_sample_size, dim)
        return sample_cov * bw_factor**2 * self.scaling
    
    def fit(self, X: pd.DataFrame, w: np.ndarray):
        if len(X) == 0:
            raise NotEnoughParticles("Fitting not possible.")
        if isinstance(X, pd.DataFrame):
            self._X_arr = X.values
        else:
            self._X_arr = X

        cov = self.fit_cov(self._X_arr, w)
        self.cov = cov
        self.normal = st.multivariate_normal(cov=self.cov, allow_singular=True)

    def rvs(self, size=None):
        arr = np.arange(len(self.X))
        sample_ind = np.random.choice(arr, size=size, p=self.w, replace=True)
        sample = self.X.iloc[sample_ind]
        perturbed = (sample +
                     np.random.multivariate_normal(
                         np.zeros(self.cov.shape[0]), self.cov,
                         size=size))
        return perturbed


    def rvs_single(self):
        perturbed = self.rvs(size=1)
        return perturbed

    def pdf(self, x: Union[pd.Series, pd.DataFrame]):
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x = x[self.X.columns]
            x = x.values
        x = np.atleast_3d(x)

        return self.pdf_static(
            x,
            self._X_arr,
            self.cov,
            self.w)

    @staticmethod
    def pdf_static(x, mean, cov, weights):
        x = np.atleast_3d(x)

        dens = (
            st.multivariate_normal(cov=cov, allow_singular=True).pdf(
                np.swapaxes(x-mean.T, 1,2))
                * weights)
        dens = np.atleast_2d(dens).sum(axis=1).squeeze()
        if dens.size == 1:
            return dens.item()
        return dens
