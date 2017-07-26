# import os
# os.chdir('/home/vnanumyan/SG/scripts/gHypE/tests')
from __future__ import division
from context import ghypeon

import ghypeon as gh
import ghypeon.biasedurn as bu
import numpy as np


def test_urn(self):
    adj = np.array([0, 3, 1])
    possib = np.array([5, 3, 4])
    return bu.dMWNCHypergeo(adj, possib)

gh.Ensemble.test_urn = test_urn

ens = gh.Ensemble(possibility=[[2, 15, 3],
                               [3, 6, 30],
                               [20, 2, 3]],
                  propensity=[[1, 2, 3],
                              [1, 2, 3],
                              [1, 2, 3]],
                  num_inter=4)


print("ensemble\t %s" % ens)
print("# nodes\t\t %s" % ens.nodes)
print("# interactions\t %s" % ens.num_inter)
print("omega\n%s" % ens.get_propensity_matrix())
print("test BUrn\t %s" %ens.test_urn())
print("")

ens.check_consistency()

ens1 = ens
print(ens1)
ens1.num_inter = 5
print("(0) # interactions\t %s" % ens.num_inter)
print("(1) # interactions\t %s" % ens1.num_inter)
print("")

ens2 = ens.copy()
print(ens2)
ens2.num_inter = 10
print("(0) # interactions\t %s" % ens.num_inter)
print("(2) # interactions\t %s" % ens2.num_inter)
print("")
#print(bu.dMWNCHypergeo(np.array([1,2]), np.array([3,4])))

adj = np.array([[0, 10, 1],
                [0, 2, 20],
                [5, 0, 0]])

print("=====================")
print("A TEST ADJACENCY:\n%s" % adj)
print("")

print("configuration possibity matrix:\n%s" % \
      ens.configuration_possibility(adj.sum(1), adj.sum(0)))
print('')

ens = gh.Ensemble()
ens.from_adjacency(adj)
ens1 = gh.ensemble_from_adjacency(adj)
print(ens)
print('Two ensembles are equal:\t %s' % (ens == ens1))
print('')

print("fit propensity\n%s" % ens.fit_propensity(adj))
print("fit propensity inplace\t %s" % ens.fit_propensity(adj, inplace=True))
print("propensity\n%s" % ens.get_propensity_matrix())
print('')

ens.replacement = True
print("replacement=True:\t%s" % (ens.replacement is True))
print("fit propensity w/ replacement\n%s" % ens.fit_propensity(adj))
