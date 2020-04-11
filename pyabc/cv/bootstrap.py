import numpy as np
from scipy import stats as st
import pandas as pd
import copy
import logging
import dask.delayed
import dask.array as da


logger = logging.getLogger(__name__)

def calc_variation(per_model_w, n_per_model, test_w):

    variations_at_X = np.stack([st.variation(ws, axis=0) for ws in per_model_w])

    
    # calculate the cv of the bootstrapped weights for each model
    #variations_at_X = np.asarray([st.variation(ws, axis=0) for ws in per_model_w])

    # normalize by number of samples per model
    model_weighted_variations_at_X = (
        variations_at_X * n_per_model[:, np.newaxis] / np.sum(n_per_model))
    

    # weight cvs by the point weights
    point_weighted_var_at_X = model_weighted_variations_at_X * test_w

    # compute an "average coefficient of variation":
    # for each model, sum up the weighted cvs over the test points
    # then, take the sum over all models
    cv = point_weighted_var_at_X.sum()

    return cv

def calc_cv_per_model(nr_particles, model_weights, N_BOOTSTR, test_w,
                  transitions, test_X):
    """
    Calculate the Coefficient of Variation.

    Parameters
    ----------

    nr_particles: int
        Number of particles to estimate the CV for

    model_weights: np.ndarray
        array of model weights

    N_BOOTSTR: int
        Nr of bootstrapped KDEs to take to estimate the CV

    test_w: List[np.ndarray]
        test_w[m] are the weights of the test points test_X[m] of model m

    transitions: List[Transition]
        List of transitions

    test_X: List[np.ndarray]
        test_X[m] are the test points with weights test_w[m]

    client: Client to execute on

    Returns
    -------

    cv, variations_at_X: float, List[np.ndarray]
        * cv is the mean variation
        * variations_at_X are the variations at the test_X

    """
    test_transitions = copy.deepcopy(transitions)

    # how many particles to draw for each model
    n_per_model = np.random.multinomial(nr_particles, model_weights)

    # N_BOOTSTR times, train test_transitions on n_per_model points, and
    # calculate the weights associated with test_X, for each model

    logger.debug("Start CV")
    futures = []
    for _ in range(N_BOOTSTR):
        futures.append(
            weights(n_per_model, transitions, test_transitions,
                    test_X))
                          
    logger.debug("Gathering futures")
    secede()
    chunked = client.gather(futures)
    rejoin()
    bootstr_w_at_test_X = [
        np.concatenate(chunk) for bs in chunked for chunk in bs]
    del chunked 
    per_model_w = [np.asarray(arr) for arr in zip(*bootstr_w_at_test_X)]

    # calculate the cv of the bootstrapped weights for each model
    variations_at_X = [st.variation(ws, axis=0) for ws in per_model_w]

    # normalize by number of samples per model
    model_weighted_variations_at_X = [
        var * n / n_per_model.sum() for
        var, n in zip(variations_at_X, n_per_model)
    ]

    # weight cvs by the point weights
    point_weighted_var_at_X = [var * w for var, w in
                               zip(model_weighted_variations_at_X, test_w)]

    # compute an "average coefficient of variation":
    # for each model, sum up the weighted cvs over the test points
    # then, take the sum over all models
    cv = sum(var.sum() for var in point_weighted_var_at_X)

    logger.debug("CV done")
    return float(cv), variations_at_X
def weights(n_per_model, transitions, test_transitions, test_X):
    """
    For each model m, sample `n_per_model[m]` points from `transitions[m]`,
    fit `test_transitions[m]` to them, and then compute the weights of all
    test points in `test_X[m]` for `test_transitions[m]`.
    Return those weights.

    Parameters
    ----------

    n_per_model: np.ndarray
        Number of samples per model

    transitions: List[Transition]
        List of transitions used to sample the bootstrapped points

    test_transitions: List[Transition]
        List of transitions used to fit to the bootstrapped points

    test_X: List[np.ndarray]
        Test points at which to evaluate the bootstrapped fitted KDEs

    Returns
    -------

    bootstr_w_at_test_X: List[np.ndarray]
        Weights for each model at the test points
    """

    from dask.distributed import get_client, secede, rejoin
    client = get_client()
    def _func(data):
        trans, test_trans, n, X = data
        bootstr_X = trans.rvs(size=n)
        test_trans.fit(bootstr_X, np.ones(len(bootstr_X)) / len(bootstr_X))
        return test_trans.pdf(X)

    futures = []
    for trans, test_trans, n, X in zip(
        transitions, test_transitions, n_per_model, test_X):
        
        # chunks of 100 per model
        chunk_size = 50
        n_chunks = int(np.ceil(n / chunk_size))
        chunked_futures = []
        for i_chunk in range(n_chunks):
            if i_chunk == n_chunks-1:
                this_chunk_size = n- (i_chunk)*chunk_size
            else:
                this_chunk_size = chunk_size
            fut = client.submit(
                _func, [trans, test_trans, this_chunk_size, X], pure=False)
            chunked_futures.append(fut)
               
        futures.append(chunked_futures)
    return futures

def calc_cv(nr_particles, model_weights, N_BOOTSTR, test_w,
                  transitions, test_X):
    """
    Calculate the Coefficient of Variation.

    Parameters
    ----------

    nr_particles: int
        Number of particles to estimate the CV for

    model_weights: np.ndarray
        array of model weights

    N_BOOTSTR: int
        Nr of bootstrapped KDEs to take to estimate the CV

    test_w: List[np.ndarray]
        test_w[m] are the weights of the test points test_X[m] of model m

    transitions: List[Transition]
        List of transitions

    test_X: List[np.ndarray]
        test_X[m] are the test points with weights test_w[m]

    client: Client to execute on

    Returns
    -------

    cv, variations_at_X: float, List[np.ndarray]
        * cv is the mean variation
        * variations_at_X are the variations at the test_X

    """
    from dask.distributed import get_client, secede, rejoin
    client = get_client()
    # create deep copies of the transitions which will be refitted
    test_transitions = copy.deepcopy(transitions)

    # how many particles to draw for each model
    n_per_model = np.random.multinomial(nr_particles, model_weights)

    # N_BOOTSTR times, train test_transitions on n_per_model points, and
    # calculate the weights associated with test_X, for each model

    logger.debug("Start CV")
    futures = []
    for _ in range(N_BOOTSTR):
        futures.append(
            weights(n_per_model, transitions, test_transitions,
                    test_X))
                          
    logger.debug("Gathering futures")
    secede()
    chunked = client.gather(futures)
    rejoin()
    bootstr_w_at_test_X = [
        np.concatenate(chunk) for bs in chunked for chunk in bs]
    del chunked 
    per_model_w = [np.asarray(arr) for arr in zip(*bootstr_w_at_test_X)]

    # calculate the cv of the bootstrapped weights for each model
    variations_at_X = [st.variation(ws, axis=0) for ws in per_model_w]

    # normalize by number of samples per model
    model_weighted_variations_at_X = [
        var * n / n_per_model.sum() for
        var, n in zip(variations_at_X, n_per_model)
    ]

    # weight cvs by the point weights
    point_weighted_var_at_X = [var * w for var, w in
                               zip(model_weighted_variations_at_X, test_w)]

    # compute an "average coefficient of variation":
    # for each model, sum up the weighted cvs over the test points
    # then, take the sum over all models
    cv = sum(var.sum() for var in point_weighted_var_at_X)

    logger.debug("CV done")
    return float(cv), variations_at_X
