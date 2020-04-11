"""
Population strategy
===================

Strategies to choose the population size.

The population size can be constant or can change over the course
of the generations.
"""

from abc import ABC, abstractmethod
import copy
import json
import logging
import numpy as np
from typing import Dict, List, Union
import warnings
from memory_profiler import profile
from dask.distributed import Client, LocalCluster
import dask.array as da
from pyabc.cv.bootstrap import calc_cv, calc_variation
from pyabc.cv.powerlaw import fitpowerlaw
from tlz import partition_all

from .transition import Transition
from collections import namedtuple

logger = logging.getLogger("Adaptation")
CVEstimate = namedtuple("CVEstimate", "n_estimated n_samples_list cvs f popt")


class PopulationStrategy(ABC):
    """
    Strategy to select the sizes of the populations.

    This is a non-functional abstract base implementation. Do not use this
    class directly. Subclasses must override the `update` method.

    Parameters
    ----------
    nr_calibration_particles:
        Number of calibration particles.
    nr_samples_per_parameter:
        Number of samples to draw for a proposed parameter.
        Default is 1.
    """

    def __init__(self,
                 nr_calibration_particles: int = None,
                 nr_samples_per_parameter: int = 1):
        self.nr_calibration_particles = nr_calibration_particles
        if nr_samples_per_parameter != 1:
            warnings.warn(
                "A nr_samples_per_parameter != 1 is deprecated "
                "since version 0.9.23, the parameter will be removed "
                "in a future release.", DeprecationWarning)
        self.nr_samples_per_parameter = nr_samples_per_parameter

    def update(self, transitions: List[Transition],
               model_weights: np.ndarray, t: int = None):
        """
        Select the population size for the next population.

        Parameters
        ----------
        transitions:
            List of transitions.
        model_weights:
            Array of model weights.
        t:
            Time to adapt for.
        """

    @abstractmethod
    def __call__(self, t: int = None) -> int:
        raise NotImplementedError()

    def get_config(self) -> dict:
        """
        Get the configuration of this object.

        Returns
        -------
        config:
            Configuration of the class as dictionary
        """
        return {"name": self.__class__.__name__,
                "nr_calibration_particles": self.nr_calibration_particles,
                "nr_samples_per_parameter": self.nr_samples_per_parameter}

    def to_json(self) -> str:
        """
        Return the configuration as json string.
        Per default, this converts the dictionary returned
        by get_config to json.

        Returns
        -------
        config:
            Configuration of the class as json string.
        """
        return json.dumps(self.get_config())


class ConstantPopulationSize(PopulationStrategy):
    """
    Constant size of the different populations

    Parameters
    ----------
    nr_particles:
        Number of particles per population.
    nr_calibration_particles:
        Number of calibration particles.
    nr_samples_per_parameter:
        Number of samples to draw for a proposed parameter.
    """

    def __init__(self,
                 nr_particles: int,
                 nr_calibration_particles: int = None,
                 nr_samples_per_parameter: int = 1):
        super().__init__(
            nr_calibration_particles=nr_calibration_particles,
            nr_samples_per_parameter=nr_samples_per_parameter)
        self.nr_particles = nr_particles

    def __call__(self, t: int = None) -> int:
        if t == -1 and self.nr_calibration_particles is not None:
            return self.nr_calibration_particles
        return self.nr_particles

    def get_config(self) -> dict:
        config = super().get_config()
        config["nr_particles"] = self.nr_particles
        return config



class AdaptivePopulationSize(PopulationStrategy):
    """
    Adapt the population size according to the mean coefficient of variation
    error criterion, as detailed in [#klingerhasenaueradaptive]_ .
    This strategy tries to respond to the shape of the
    current posterior approximation by selecting the population size such
    that the variation of the density estimates matches the target
    variation given via the mean_cv argument.

    Parameters
    ----------
    start_nr_particles:
        Number of particles in the first populations
    mean_cv:
        The error criterion. Defaults to 0.05.
        A smaller value leads generally to larger populations.
        The error criterion is the mean coefficient of variation of
        the estimated KDE.
    max_population_size:
        Max nr of allowed particles in a population.
        Defaults to infinity.
    min_population_size:
        Min number of particles allowed in a population.
        Defaults to 10
    nr_samples_per_parameter:
        Defaults to 1.
    n_bootstrap:
        Number of bootstrapped populations to use to estimate the CV.
        Defaults to 10.
    nr_calibration_particles:
        Number of calibration particles.


    .. [#klingerhasenaueradaptive] Klinger, Emmanuel, and Jan Hasenauer.
            “A Scheme for Adaptive Selection of Population Sizes in "
            Approximate Bayesian Computation - Sequential Monte Carlo."
            Computational Methods in Systems Biology, 128-44.
            Lecture Notes in Computer Science.
            Springer, Cham, 2017.
            https://doi.org/10.1007/978-3-319-67471-1_8.
    """

    def __init__(self,
                 start_nr_particles,
                 mean_cv: float = 0.05,
                 max_population_size: int = np.inf,
                 min_population_size: int = 10,
                 nr_samples_per_parameter: int = 1,
                 n_bootstrap: int = 10,
                 nr_calibration_particles: int = None,
                 client=None):
        super().__init__(
            nr_calibration_particles=nr_calibration_particles,
            nr_samples_per_parameter=nr_samples_per_parameter)

        if client is None:
            logger.info("Using local client")
            cluster = LocalCluster(n_workers=1)
            client = Client(cluster)
        self.client = client 

        self.start_nr_particles = start_nr_particles
        self.max_population_size = max_population_size
        self.min_population_size = min_population_size
        self.mean_cv = mean_cv
        self.n_bootstrap = n_bootstrap

        # to hold the current value
        self.nr_particles = start_nr_particles

    def get_config(self) -> dict:
        config = super().get_config()
        config["start_nr_particles"] = self.start_nr_particles
        config["max_population_size"] = self.max_population_size
        config["min_population_size"] = self.min_population_size
        config["mean_cv"] = self.mean_cv
        config["n_bootstrap"] = self.n_bootstrap
        return config

    def update(self, transitions: List[Transition],
               model_weights: np.ndarray, t: int = None):

        cv_estimate = self.predict_population_size(
            model_weights, transitions)

        reference_nr_part = self.nr_particles
        if not np.isnan(cv_estimate.n_estimated):
            self.nr_particles = max(min(int(cv_estimate.n_estimated),
                                        self.max_population_size),
                                    self.min_population_size)

        logger.info("Change nr particles {} -> {}"
                    .format(reference_nr_part, self.nr_particles))

    def __call__(self, t: int = None) -> int:
        if t == -1 and self.nr_calibration_particles is not None:
            return self.nr_calibration_particles
        return self.nr_particles



    # @profile
    def predict_population_size(
            self,
            model_weights,
            transitions,
            n_steps=10,
            first_step_factor=3) -> CVEstimate:
        """
        Estimate the required nr of particles for a target coefficient of
        variation

        Parameters
        ----------

        TODO


        n_steps: int
            The number of steps

        first_step_factor: float
            Factor by which to divide the current population size, to give the
            lower bound for the next population size.

        Returns
        -------

        suggested_pop_size: int
        """
        test_Xs = [trans.X for trans in transitions]

        # n_models in first dimension
        test_w = np.vstack([trans.w for trans in transitions])
        
        current_pop_size = self.nr_particles
        target_cv = self.mean_cv

        if current_pop_size == 1:
            return CVEstimate(1, [], [], None, None)

        start = max(current_pop_size // first_step_factor, 1)
        stop = current_pop_size * 2
        step = max(current_pop_size // n_steps, 1)

        n_samples_list = list(range(start, stop, step))
        
        
        per_model_weights = []
        

        n_samples_futures = [] 
        n_per_models = []
        cvs = []


        br_size = test_Xs[0].shape[0] * test_Xs[0].shape[1] * stop * 64 / 1000 / 1000
        target_size = 100 # MiB

        n_chunks = int(np.ceil(br_size / target_size))
        chunk_size = int(np.ceil(test_Xs[0].shape[0] / n_chunks))
        print("Chunk size: ", chunk_size)
        test_X_arrs = [da.from_array(
            test_X.values, chunks=(chunk_size, test_X.shape[1])) for test_X in test_Xs]
        
        # test_X_arrs = self.client.scatter(
        #     test_X_arrs)
        
        test_transitions_settings = [
            (tr.bandwidth_selector, tr.scaling) for tr in transitions]
        
        for i, ns in enumerate(n_samples_list):
            n_per_model = np.random.multinomial(ns, model_weights)
            n_per_models.append(n_per_model)
            model_futures = []
            for j, (n, transition, test_X) in enumerate(zip(
                    n_per_model, transitions, test_X_arrs)):
                # test_X = self.client.scatter(test_X, broadcast=True)
                bootstrap_futures = []
                
                for _ in range(self.n_bootstrap):
                    
                    bootstr_X = transition.rvs(size=n).values
                    weights_init = np.ones(len(bootstr_X)) / len(bootstr_X)
                    cov = transition.fit_cov(
                        bootstr_X,
                        weights_init
                        )
                    bootstrap = da.map_blocks(
                        transition.pdf_static,
                        test_X,
                        bootstr_X,
                        cov,
                        weights_init,
                        drop_axis=1,
                        dtype=float)
                    bootstrap = self.client.compute(bootstrap) 
                    bootstrap_futures.append(bootstrap)
                bootstrap_futures = self.client.gather(bootstrap_futures)
                model_futures.append(bootstrap_futures)

            cv = calc_variation(
                    model_futures,
                    n_per_model,
                    test_w)
                    
            cvs.append(cv)
                
        # cvs_array = da.stack(cvs)
        # cvs = cvs_array.compute()
                    
        try:
            popt, f, finv = fitpowerlaw(n_samples_list, cvs)
            suggested_pop_size = finv(target_cv)
            return CVEstimate(suggested_pop_size, n_samples_list, cvs, f, popt)
        except RuntimeError:
            logger.warning("Power law fit failed. "
                           "Falling back to current nr particles {}"
                           .format(current_pop_size))
            return CVEstimate(current_pop_size, n_samples_list, cvs, None, None)

class ListPopulationSize(PopulationStrategy):
    """
    Return population size values from a predefined list. For every time point
    enquired later (specified by time t), an entry must exist in the list.

    Parameters
    ----------
    values: List[float]
        List of population size values.
        ``values[t]`` is the value for population t.
    nr_calibration_particles:
        Number of calibration particles.
    """

    def __init__(self,
                 values: Union[List[int], Dict[int, int]],
                 nr_calibration_particles: int = None,
                 nr_samples_per_parameter: int = 1):
        super().__init__(
            nr_calibration_particles=nr_calibration_particles,
            nr_samples_per_parameter=nr_samples_per_parameter)
        self.values = values

    def get_config(self) -> dict:
        config = super().get_config()
        config["population_values"] = self.population_values
        return config

    def __call__(self, t: int = None) -> int:
        if t == -1 and self.nr_calibration_particles is not None:
            return self.nr_calibration_particles
        return self.values[t]
