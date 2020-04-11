from abc import abstractmethod
from typing import Union
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from .exceptions import NotEnoughParticles
from .transitionmeta import TransitionMeta

logger = logging.getLogger("Transitions")


class Transition(BaseEstimator, metaclass=TransitionMeta):
    """
    Abstract Transition base class. Derive all Transitions from this class

        .. note::
            This class does a little bit of meta-programming.

            The `fit`, `pdf` and `rvs` methods are automatically wrapped
            to handle the special case of no parameters.

            Hence, you can safely assume that you encounter at least one
            parameter. All the defined transitions will then automatically
            generalize to the case of no paramter.
    """
    NR_BOOTSTRAP = 5
    X = None
    w = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, w: np.ndarray):
        """
        Fit the density estimator (perturber) to the sampled data.
        Concrete implementations might do something like fitting a KDE.

        The parameters given as ``X`` and ``w`` are automatically stored
        in ``self.X`` and ``self.w``.

        Parameters
        ----------
        X: pd.DataFrame
            The parameters.
        w: array
            The corresponding weights
        """

    @abstractmethod
    def rvs_single(self) -> pd.Series:
        """
        Random variable sample (rvs).

        Sample from the fitted distribution.

        Returns
        -------
        sample: pd.Series
            A sample from the fitted model.
        """

    def rvs(self, size=None):
        """
        Sample from the density.

        Parameters
        ----------

        size: int, optional
            Number of independent samples to draw.
            Defaults to 1 and is in this case equivalent to calling
            "rvs_single".

        Returns
        -------

        samples: The samples as pandas DataFrame


        Note
        ----

        This method can be overridden for efficient implementations.
        The default is to call rvs_single repeatedly (which might
        not be the most efficient way).

        """
        if size is None:
            return self.rvs_single()
        return pd.DataFrame([self.rvs_single() for _ in range(size)])

    @abstractmethod
    def pdf(self, x: Union[pd.Series, pd.DataFrame]) \
            -> Union[float, np.ndarray]:
        """
        Evaluate the probability density function (PDF) at `x`.

        Parameters
        ----------
        x: pd.Series, pd.DataFrame
            Parameter. If x is a series, then x should have the the columns
            from X passed to the fit method as indices.
            If x is a DataFrame, then x should have the same columns as X
            passed before to the fit method. The order of the columns is not
            important

        Returns
        -------

        density: float
            Probability density at `x`.
        """

    def score(self, X: pd.DataFrame, w: np.ndarray):
        densities = self.pdf(X)
        return (np.log(densities) * w).sum()

    def no_meaningful_particles(self) -> bool:
        return len(self.X) == 0 or self.no_parameters

class DiscreteTransition(Transition):
    """
    This is a base class for discrete transition kernels.
    """
