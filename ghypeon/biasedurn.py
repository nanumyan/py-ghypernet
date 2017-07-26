#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:37:36 2017

@author: vnanumyan
"""

from __future__ import division

from rpy2.robjects.packages import importr
#from rpy2 import robjects
#from rpy2.robjects import FloatVector, IntVector, Array
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from numpy import ndarray


_rbiasedUrn = importr('BiasedUrn')


def dMWNCHypergeo(adj, xi, omega=None):
    """
    Wrap the R implementation of BiasedUrn in python using `rpy2`
    
    This implementation is to be changed to a native Python binding
    to the C++ implementation of hypergeometric functions
    
    Args:
        adj (ndarray): a 1D array of length `n` with integer observations
            each of an `n`-dimensional variable 
        xi (ndarray): a 1D array of length `n` with each element setting 
            the maximum of the corresponding dimension of the variable
        omega (:obj:`ndarray`, optional): a 1D array of length `n` providing the 
            arbitrarily scaled odds for each dimension of the variable.
            If not provided, all odds are equal and the distribution is 
            *Multivariate Hypergeometric*
    Returns:
        float: The probability to observe `adj`, given `xi` and `omega`
    """
    
    assert type(adj) is ndarray, "`adj` must be a numpy.array"
    assert type(xi) is ndarray, "`xi` must be a numpy.array"
    assert adj.shape == xi.shape, "shapes of `adj` and `xi` don't match"
    if omega:
        assert type(omega) is ndarray, "`omega` must be a numpy.array"
        assert omega.shape == xi.shape, "shapes of `omega` and `xi` don't match"
    else:
        omega = 1
    m = adj.sum()
    
    return _rbiasedUrn.dMWNCHypergeo(x=adj, m=xi, n=m, odds=omega)[0]