#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Gibbs sampler for a state-ful model, using the MH code.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["GibbsSampler", "GibbsSubController", "GibbsController"]

import numpy as np

from . import autocorr
from .sampler import Sampler


# === MHSampler ===
class GibbsSampler(Sampler):
    """
    The most basic possible Metropolis-Hastings style MCMC sampler, designed to be 
    encapsulated as part of a state-ful Gibbs sampler.

    :param cov:
        The covariance matrix to use for the proposal distribution.

    :param dim:
        Number of dimensions in the parameter space.

    :param lnpostfn:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the posterior probability for that
        position.

    :param revertfn:
        A function which reverts the model to the previous parameter values, in the
        case that the proposal parameters are rejected.

    :param args: (optional)
        A list of extra positional arguments for ``lnpostfn``. ``lnpostfn``
        will be called with the sequence ``lnpostfn(p, *args, **kwargs)``.

    :param kwargs: (optional)
        A list of extra keyword arguments for ``lnpostfn``. ``lnpostfn``
        will be called with the sequence ``lnpostfn(p, *args, **kwargs)``.

    """
    def __init__(self, cov, p0, revertfn, *args, **kwargs):
        try:
            self.debug = kwargs["debug"]
        except KeyError:
            self.debug = False
        super(GibbsSampler, self).__init__(*args, **kwargs)
        self.cov = cov
        self.p0 = p0
        self.revertfn = revertfn

    def reset(self):
        super(GibbsSampler, self).reset()
        self._chain = np.empty((0, self.dim))
        self._lnprob = np.empty(0)

    def sample(self, p0, lnprob0=None, randomstate=None, thin=1,
               storechain=True, iterations=1):
        """
        Advances the chain ``iterations`` steps as an iterator

        :param p0:
            The initial position vector.

        :param lnprob0: (optional)
            The log posterior probability at position ``p0``. If ``lnprob``
            is not provided, the initial value is calculated.

        :param rstate0: (optional)
            The state of the random number generator. See the
            :func:`random_state` property for details.

        :param iterations: (optional)
            The number of steps to run.

        :param thin: (optional)
            If you only want to store and yield every ``thin`` samples in the
            chain, set thin to an integer greater than 1.

        :param storechain: (optional)
            By default, the sampler stores (in memory) the positions and
            log-probabilities of the samples in the chain. If you are
            using another method to store the samples to a file or if you
            don't need to analyse the samples after the fact (for burn-in
            for example) set ``storechain`` to ``False``.

        At each iteration, this generator yields:

        * ``pos`` - The current positions of the chain in the parameter
          space.

        * ``lnprob`` - The value of the log posterior at ``pos`` .

        * ``rstate`` - The current state of the random number generator.

        """

        self.random_state = randomstate

        p = np.array(p0)
        if lnprob0 is None:
            lnprob0 = self.get_lnprob(p)

        # Resize the chain in advance.
        if storechain:
            N = int(iterations / thin)
            self._chain = np.concatenate((self._chain,
                                          np.zeros((N, self.dim))), axis=0)
            self._lnprob = np.append(self._lnprob, np.zeros(N))

        i0 = self.iterations
        # Use range instead of xrange for python 3 compatability
        for i in range(int(iterations)):
            self.iterations += 1

            # Calculate the proposal distribution.
            if self.dim == 1:
                q = self._random.normal(loc=p[0], scale=self.cov[0], size=(1,))
            else:
                q = self._random.multivariate_normal(p, self.cov)

            newlnprob = self.get_lnprob(q)
            diff = newlnprob - lnprob0
            if self.debug:
                print("old lnprob: {}".format(lnprob0))
                print("proposed lnprob: {}".format(newlnprob))

            # M-H acceptance ratio
            if diff < 0:
                diff = np.exp(diff) - self._random.rand()
                if diff < 0:
                    #Reject the proposal and revert the state of the model
                    if self.debug:
                        print("Proposal rejected")
                    self.revertfn()

            if diff > 0:
                #Accept the new proposal
                if self.debug:
                    print("Proposal accepted")
                p = q
                lnprob0 = newlnprob
                self.naccepted += 1

            if storechain and i % thin == 0:
                ind = i0 + int(i / thin)
                self._chain[ind, :] = p
                self._lnprob[ind] = lnprob0

            # Heavy duty iterator action going on right here...
            yield p, lnprob0, self.random_state

    @property
    def acor(self):
        """
        An estimate of the autocorrelation time for each parameter (length:
        ``dim``).

        """
        return self.get_autocorr_time()

    def get_autocorr_time(self, window=50):
        """
        Compute an estimate of the autocorrelation time for each parameter
        (length: ``dim``).

        :param window: (optional)
            The size of the windowing function. This is equivalent to the
            maximum number of lags to use. (default: 50)

        """
        return autocorr.integrated_time(self.chain, axis=0, window=window)

class GibbsSubController:
    '''
    Designed to be a node in the hierarchy of samplers, yet still be run by a GibbsController

    :param samplers: a list of the various GibbsSampler or GibbsSubController objects to iterate amongst

    '''

    def __init__(self, samplers, **kwargs):
        self.samplers = samplers
        self.nsamplers = len(self.samplers)
        self.debug = kwargs.get("debug", False)
        self.p0 = None

    def reset(self):
        for sampler in self.samplers:
            sampler.reset()

    def run_mcmc(self, p0, iterations, lnprob0):
        #p0 is taken as a parameter to act like a GibbsSampler for compatibility with GibbsController but is discarded
        for i in range(iterations):
            if self.debug:
                print("\n\nGibbsSubController on iteration {} of {}".format(i, iterations))
                print("there are {} samplers".format(self.samplers))
            for sampler in self.samplers:
                if self.debug:
                    print("on sampler", sampler)
                sampler.p0, lnprob0, state = sampler.run_mcmc(sampler.p0, 1, lnprob0=lnprob0)
                if self.debug:
                    print("lnprob0 is", lnprob0)
        return (None, lnprob0, None)

    @property
    def acceptance_fraction(self):
        return [sampler.acceptance_fraction for sampler in self.samplers]

    @property
    def acor(self):
        return [sampler.acor for sampler in self.samplers]

    @property
    def flatchain(self):
        '''
        Stack all of the separate subsamples into the standard emcee format. Assumes that each subsampler
        has been run for the same amount of iterations.
        '''
        #check to make sure each sampler has been run for the same amount of iterations
        assert np.all([sampler.iterations == self.samplers[0].iterations for sampler in self.samplers])
        #concatenate samples into standard emcee format
        return np.hstack([sampler.flatchain for sampler in self.samplers])

class GibbsController:
    '''
    One Sampler to rule them all. Rotates among all of the GibbsSamplers and GibbsSubControllers indiscriminately.

    :param samplers: a list of the various GibbsSampler or GibbsSubController objects to iterate amongst
    '''

    def __init__(self, samplers, **kwargs):
        self.samplers = samplers
        self.nsamplers = len(self.samplers)
        self.debug = kwargs.get("debug", False)

    def reset(self):
        for sampler in self.samplers:
            sampler.reset()

    def run(self, iterations):
        lnprob0 = -np.inf
        for i in range(iterations):
            if self.debug:
                print("\n\nGibbsController on iteration {} of {}".format(i, iterations))
            for sampler in self.samplers:
                sampler.p0, lnprob0, state = sampler.run_mcmc(sampler.p0, 1, lnprob0=lnprob0)
                if self.debug:
                    print("lnprob0 is", lnprob0)

    @property
    def acceptance_fraction(self):
        return [sampler.acceptance_fraction for sampler in self.samplers]

    @property
    def acor(self):
        return [sampler.acor for sampler in self.samplers]

    @property
    def flatchain(self):
        '''
        Stack all of the separate subsamples into the standard emcee format. Assumes that each subsampler
        has been run for the same amount of iterations.
        '''
        #check to make sure each sampler has been run for the same amount of iterations
        assert np.all([sampler.iterations == self.samplers[0].iterations for sampler in self.samplers])
        #concatenate samples into standard emcee format
        return np.hstack([sampler.flatchain for sampler in self.samplers])
