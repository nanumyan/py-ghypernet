#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:37:36 2017

@author: vnanumyan
"""

from __future__ import division
from copy import deepcopy
from pprint import pformat
# import ghypeon.biasedurn as _bu
import numpy as _np


## TODO: ADD LOGGING, CHANGE CORRESPONDING `PRINT`S

class Ensemble(object):
    """
    A class implementing the Generalized Hypergeometric Network Ensembles.
    An instance of this class represents an ensemble with certain
    "edge propensities" and "edge possibilities".
    """

    def __init__(self, possibility=None, propensity=None, num_inter=None,
                 directed=True, selfloops=True, replacement=False):
        """
        Generate an ensemble of networks according to the
        Generalized Hypergeometric Network Ensembles

        Args:
            possibility (ndarray):
                `NxM` dimensional array with integer possibilities, i.e.,
                the maximum number of observations for each pair
                $n\\in[0,N-1], m\\in[0,M-1]$
            propensity (:obj:`ndarray`, optional):
                `NxM` dimensional array with float-valued odds for each pair
                $n\\in[0,N-1], m\\in[0,M-1]$.
                It is recommended to normalize these to 1 (BiasedUrn R package)
            num_inter (int):
                The number of interactions in the observed or realized networks
            directed (bool):
                The directedness of the network.
                If `False`, only the upper triangles of `possibility` and
                `propensity` will be taken.
                Defaults to `True`.

                TODO: Should be valid only for unipartite networks (N=M)
            selfloops (bool):
                If `True`, self-loops are allowed, i.e. diagonal elements in
                `possibility` and `propensity`
            (REMOVED) biased (bool, optional):
                If `propensity` should be considered or not.
                Equivalent to the soft configuration model and `propensity=const`
                Defaults to `None`.
            replacement (bool):
                Defaults to `False`.
                If the drawing of elements from `possibility` are done with replacement.
                If `True`, the distribution is of hypergeometric type.
                If `False`, it is multinomial.
        """

        # set the number of interactions
        self.num_inter = num_inter

        # set the directedness
        self.directed = directed

        # set if the self loops are allowed
        self.selfloops = selfloops

        # set replacement
        self.replacement = replacement

        # set the number of nodes
        self.nodes = None

        # set the possibility matrix
        self.possibility = None
        if possibility:
            # sets `possibility`, returns the shape of `possibility`
            self.set_possibility(possibility)
            self.__set_nodes(possibility)

        # set the propensities
        if propensity is None:
            #    self.biased = False
            self.propensity = None
        else:
            #    self.biased = True
            self.set_propensity(propensity)

        # RUN A TEST FOR CONSISTENCY AMONG PARAMETERS
        self.check_consistency()

    ### END _init__
    ###--------------------------------------

    def __set_nodes(self, matrix):
        """
        Infers and sets the number of nodes from the shape of the `matrix`,
        which should be the possibility, or adjacency
        """
        shape = _np.asarray(matrix).shape
        if shape[0] == shape[1]:
            self.nodes = shape[0]
        else:
            err = "{}x{} matrix: bipartite networks not supported yet"
            raise NotImplementedError(err.format(shape[0], shape[1]))
    ### END set_nodes
    ###--------------------------------------

    def set_possibility(self, possibility):
        """
        set the instance `possibility` from an input array, or list, or matrix,
        while accounting for directedness and selfloops.
        """
        possibility = _np.asarray(possibility)
        self.possibility = self.__vectorize_matrix(possibility)

    ### END set_possibility
    ###--------------------------------------

    def set_propensity(self, propensity):
        """
        set the instance `propensity` from an input array, or list, or matrix,
        while accounting for directedness and selfloops.
        Also normalizes `propensity` to $max(propensity)=1$
        """
        propensity = _np.asarray(propensity)
        propensity = self.__vectorize_matrix(propensity)

        assert propensity.shape == self.possibility.shape, \
            "shapes of `propensity` and `possibility` do not match"

        propensity = propensity / propensity.max()
        self.propensity = propensity

    ###--------------------------------------
    ### END set_propensity

    def check_consistency(self):
        """
        check if all attribute values are consistent with each other
        """

        if all(attr is None for attr in (self.nodes, self.num_inter,
                                         self.possibility, self.propensity)):
            return None

        nodes = self.nodes

        # check if the size of `possibility` is correct
        if self.directed:
            size = nodes * nodes
        else:
            size = (nodes * (nodes + 1)) / 2
        if not self.selfloops:
            size = size - nodes

        assert len(self.possibility) == size, \
            "number of elements in `possibility` and params of ensemble are not consistent"

        # check if `propensity` and `possibility match in shape
        if self.propensity is not None:
            assert self.propensity.shape == self.possibility.shape, \
                "shapes of `propensity` and `possibility` do not match"

        # check if the number of interactions is valid
        assert self.num_inter <= self.possibility.sum(), \
            "`num_inter` must be smaller than the sum of `possibility`"

    ### END consistency
    ###--------------------------------------

    def from_adjacency(self, adj, biased=True, directed=None, selfloops=None):
        # TODO: from_adjacency
        """
        Build the Ensemble from a given `adj` adjacency matrix.
        If `self.biased=True`, it will centered around the the given adjacency.
        Otherwise, just the possibilities according to configuration model will be set.

        Args:
            adj (ndarray):
                The adjacency matrix from which the ensemble is built
                based on the configuration model
            biased (bool):
                If `True` the maximum likelihood propensity matrix will be inferred and set
                Defaults to `True`.
            directed (bool, optional):
                Sets the directionality.
                If `False`, the upper triangle of `adj` is considered
                Defaults to `None`, in which case the directionality is inferred from `adj`
            selfloops (bool, optional):
                If self-loops are allowed in the ensemble
                If `False`, the diagonal of the `adj` is ignored, even if not empty
                Defaults to `None`, in which case it is inferred from `adj` --
                `True`, if there are non-zero diagonal elements in `adj`,
                `False` otherwise.
        """

        _adj = _np.asarray(adj)
        # set the number of nodes
        self.__set_nodes(_adj)

        # set directionality
        if isinstance(directed, bool):
            self.directed = directed
        else:
            if _np.all(_adj == _adj.T):
                self.directed = False
            else:
                self.directed = True

        # set selfloops
        if isinstance(selfloops, bool):
            self.selfloops = selfloops
        else:
            if _np.all(_adj.diagonal() == 0):
                self.selfloops = False
            else:
                self.selfloops = True

        # set # interactions
        _adj_vec = self.__vectorize_matrix(adj)
        self.num_inter = _adj_vec.sum()

        # set the possibility matrix
        self.configuration_possibility(_adj.sum(1), _adj.sum(0), inplace=True)

        # fit propensities if needed
        if biased:
            self.fit_propensity(adj, inplace=True)

        # check consistency
        self.check_consistency()

    ### END from_adjacency
    ###--------------------------------------

    def configuration_possibility(self, kout, kin=None, inplace=False):
        """
        Build the possibility matrix from the degrees.

        Args:
            kout (ndarray):
                1D array of node (out-)degrees.
                If `self.directed=False`, this is the total degree.
            kin (ndarray, optional):
                1D array of node in-degrees.
                Necessary, if `self.directed=True`.
            inplace (bool):
                If the possibilities should be set inplace.
                Defaults to `False`

        Returns:
            None: if `inplace=True`
            ndarray: the possibility matrix, if `inplace=False`
        """
        # TODO: configuration_possibilities
        if self.directed:
            if kin is None:
                raise ValueError("`kin` must be provided, if the ensemble is directed")
            possib = _np.outer(kout, kin)
        else:
            possib = _np.outer(kout, kout)

        if inplace:
            self.possibility = self.__vectorize_matrix(possib)
            return None
        return possib

    ### END configuration_possibilities
    ###--------------------------------------

    def fit_propensity(self, adj, inplace=False):
        """
        Fit the propensity matrix.
        If `self.replacement = True`, according to
        multivariate Wallenius Non-Central Hypergeometric distribution.
        Else, according to Multinomial distribution
        """
        adj_vec = self.__vectorize_matrix(adj)

        if self.replacement:
            omega = self.__fit_propensity_multinom(adj_vec)
        else:
            omega = self.__fit_propensity_wallenius(adj_vec)

        if inplace:
            self.propensity = _np.abs(omega)
            return None
        return self.__get_matrix(_np.abs(omega))

    ### END fit_propensity
    ###--------------------------------------

    def __fit_propensity_wallenius(self, adj_vec):
        """
        Compute the Maximum Likelihood Estimation (whenever possible) of the propensity matrix
        according to the multivariate Wallenius Non-Central Hypergeometric distribution.
        If the MLE does not exist, clamping is done

        TODO: explain the clamping
        """

        assert _np.all(adj_vec <= self.possibility), "more observations than possibilities"
        omega = _np.empty_like(self.possibility, dtype=_np.float)

        # clamping: find entries for which MLE does not exist
        # set these to 0s and 1s
        _idx_zero = (adj_vec == 0) & (self.possibility == 0)
        _idx_one = (adj_vec == self.possibility)
        omega[_idx_zero] = 0.0
        omega[_idx_one] = 1.0

        # get the MLE for the rest
        _idx_fit = ~(_idx_zero | _idx_one)
        omega[_idx_fit] = - _np.log(1 - adj_vec[_idx_fit] / self.possibility[_idx_fit])
        omega[_idx_fit] = omega[_idx_fit] / _np.max(omega[_idx_fit])

        # taking the absolute value helps with numerical errors around zero
        return _np.abs(omega)

    ### END __fit_propensity_wallenius
    ###--------------------------------------

    def __fit_propensity_multinom(self, adj_vec):
        """
        Compute the Maximum Likelihood Estimation (whenever possible) of the propensity matrix
        according to the Multinomial distribution.
        """

        assert _np.all(adj_vec <= self.possibility), "more observations than possibilities"
        omega = adj_vec / self.possibility

        omega = omega / _np.max(omega)

        # taking the absolute value helps with numerical errors around zero
        return _np.abs(omega)

    ### END __fit_propensity_multinom
    ###--------------------------------------

    ##########################################################################
    ## HELPER FUNCTIONS
    ##########################################################################
    def copy(self):
        """
        Return a deep copy of the ensemble.
        Useful, if a variation of the ensemble with only some attributes changed
        is needed.
        """
        return deepcopy(self)

    ### END copy
    ###--------------------------------------

    def __vectorize_matrix(self, matrix):
        """
        Flattens the `matrix` to the minimum vector with respect to
        `self.directed` and `self.selfloops`.

        Args:
            matrix (array-like): the matrix to vectorize

        Returns:
            ndarray: 1D numpy array
        """

        vectorized = _np.asarray(matrix).copy()
        if self.directed:
            if not self.selfloops:
                _np.fill_diagonal(vectorized, 0)
            vectorized = vectorized.flatten()
        else:
            if self.selfloops:
                vectorized = vectorized[_np.triu_indices_from(vectorized)]
            else:
                vectorized = vectorized[_np.triu_indices_from(vectorized, 1)]

        return vectorized

    ### END __vectorize_matrix
    ###--------------------------------------

    def get_possibility_matrix(self):
        """
        Returns:
            ndarray: the possibility matrix of the ensemble
        """
        return self.__get_matrix(self.possibility)

    def get_propensity_matrix(self):
        """
        Returns:
            ndarray: the propensity matrix of the ensemble
        """
        return self.__get_matrix(self.propensity)

    def __get_matrix(self, vectorized_matrix):
        """
        Recover the matrix form from a vectorized version that was produced
        using `__vectorize_matrix` method
        """

        if self.directed:
            matrix = vectorized_matrix.reshape((self.nodes, self.nodes))
        else:
            matrix = _np.zeros((self.nodes, self.nodes))
            if self.selfloops:
                matrix[_np.triu_indices_from(matrix)] = vectorized_matrix
            else:
                matrix[_np.triu_indices_from(matrix, 1)] = vectorized_matrix

        return matrix
        ### END __vectorize_matrix
        ###--------------------------------------
    def __str__(self):
        return pformat(self.__dict__)

    def __eq__(self, other):
        return all(_np.all(v == other.__dict__[k]) for k, v in self.__dict__.items())

    def __ne__(self, other):
        return not self == other

### END CLASS Ensemble
###--------------------------------------

def ensemble_from_adjacency(adjacency, biased=True, directed=None,
                            selfloops=None, replacement=False):
    """
    Build the Ensemble from a given `adj` adjacency matrix.
        If `self.biased=True`, it will centered around the the given adjacency.
        Otherwise, just the possibilities according to configuration model will be set.

    Args:
        adj (ndarray):
            The adjacency matrix from which the ensemble is built
            based on the configuration model
        biased (bool):
            If `True` the maximum likelihood propensity matrix will be inferred and set
            Defaults to `True`.
        directed (bool, optional):
            Sets the directionality.
            If `False`, the upper triangle of `adj` is considered
            Defaults to `None`, in which case the directionality is inferred from `adj`
        selfloops (bool, optional):
            If self-loops are allowed in the ensemble
            If `False`, the diagonal of the `adj` is ignored, even if not empty
            Defaults to `None`, in which case it is inferred from `adj` --
            `True`, if there are non-zero diagonal elements in `adj`,
            `False` otherwise.
        replacement (bool):
            Defaults to `False`.
            If the drawing of elements from `possibility` are done with replacement.
            If `True`, the distribution is of hypergeometric type.
            If `False`, it is multinomial.
    """
    ensemble = Ensemble(replacement=replacement)
    ensemble.from_adjacency(adjacency, biased=biased, directed=directed, selfloops=selfloops)

    return ensemble
