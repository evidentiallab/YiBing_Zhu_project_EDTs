"""
Mathisbing: Modifications made on someone else's code. From https://github.com/reineking/pyds
"""

from __future__ import print_function
from itertools import chain, combinations
from functools import partial, reduce
from operator import mul
from math import log, fsum, sqrt
from random import random, shuffle, uniform
import sys

try:
    import numpy
    try:
        from scipy.stats import chi2
        from scipy.optimize import fmin_cobyla
    except:
        print('SciPy not found: some features will not work.', file=sys.stderr)
except:
    print('NumPy not found: some features will not work.', file=sys.stderr)


class MassFunction(dict):
    def __init__(self, source=None):
        if source != None:
            if isinstance(source, dict):
                source = source.items()
            for (h, v) in source:
                self[h] += v
    
    @staticmethod
    def _convert(hypothesis):
        if isinstance(hypothesis, frozenset):
            return hypothesis
        else:
            return frozenset(hypothesis)
    
    @staticmethod
    def gbt(likelihoods, normalization=True, sample_count=None):
        m = MassFunction()
        if isinstance(likelihoods, dict):
            likelihoods = list(likelihoods.items())
        # filter trivial likelihoods 0 and 1
        ones = [h for (h, l) in likelihoods if l >= 1.0]
        likelihoods = [(h, l) for (h, l) in likelihoods if 0.0 < l < 1.0]
        if sample_count == None:   # deterministic
            def traverse(m, likelihoods, ones, index, hyp, mass):
                if index == len(likelihoods):
                    m[hyp + ones] = mass
                else:
                    traverse(m, likelihoods, ones, index + 1, hyp + [likelihoods[index][0]], mass * likelihoods[index][1])
                    traverse(m, likelihoods, ones, index + 1, hyp, mass * (1.0 - likelihoods[index][1]))
            traverse(m, likelihoods, ones, 0, [], 1.0)
            if normalization:
                m.normalize()
        else:   # Monte-Carlo
            if normalization:
                empty_mass = reduce(mul, [1.0 - l[1] for l in likelihoods], 1.0)
            for _ in range(sample_count):
                rv = [random() for _ in range(len(likelihoods))]
                subtree_mass = 1.0
                hyp = set(ones)
                for k in range(len(likelihoods)):
                    l = likelihoods[k][1]
                    p_t = l * subtree_mass
                    p_f = (1.0 - l) * subtree_mass
                    if normalization and not hyp: # avoid empty hypotheses in the normalized case
                        p_f -= empty_mass
                    if p_t > rv[k] * (p_t + p_f):
                        hyp.add(likelihoods[k][0])
                    else:
                        subtree_mass *= 1 - l # only relevant for the normalized empty case
                m[hyp] += 1.0 / sample_count
        return m
    
    @staticmethod
    def from_bel(bel):
        m = MassFunction()
        for h1 in bel.keys():
            v = fsum([bel[h2] * (-1)**(len(h1 - h2)) for h2 in powerset(h1)])
            if v > 0:
                m[h1] = v
        mass_sum = fsum(m.values())
        if mass_sum < 1.0:
            m[frozenset()] = 1.0 - mass_sum
        return m
    
    @staticmethod
    def from_pl(pl):
        frame = max(pl.keys(), key=len)
        bel_theta = pl[frame]
        bel = {frozenset(frame - h):bel_theta - v for (h, v) in pl.items()} # follows from bel(-A) = bel(frame) - pl(A)
        return MassFunction.from_bel(bel)
    
    @staticmethod
    def from_q(q):
        m = MassFunction()
        frame = max(q.keys(), key=len)
        for h1 in q.keys():
            v = fsum([q[h1 | h2] * (-1)**(len(h2 - h1)) for h2 in powerset(frame - h1)])
            if v > 0:
                m[h1] = v
        mass_sum = fsum(m.values())
        if mass_sum < 1.0:
            m[frozenset()] = 1.0 - mass_sum
        return m
    
    def __missing__(self, key):
        return 0.0
    
    def __copy__(self):
        c = MassFunction()
        for k, v in self.items():
            c[k] = v
        return c
    
    def copy(self):
        return self.__copy__()
    
    def __contains__(self, hypothesis):
        return dict.__contains__(self, MassFunction._convert(hypothesis))
    
    def __getitem__(self, hypothesis):
        return dict.__getitem__(self, MassFunction._convert(hypothesis))
    
    def __setitem__(self, hypothesis, value):
        if value < 0.0:
            raise ValueError("mass value is negative: %f" % value)
        dict.__setitem__(self, MassFunction._convert(hypothesis), value)
    
    def __delitem__(self, hypothesis):
        return dict.__delitem__(self, MassFunction._convert(hypothesis))
    
    def frame(self):
        if not self:
            return frozenset()
        else:
            return frozenset.union(*self.keys())
    
    def singletons(self):
        return {frozenset((s,)) for s in self.frame()}
    
    def focal(self):
        return {h for (h, v) in self.items() if v > 0}
    
    def core(self, *mass_functions):
        if mass_functions:
            return frozenset.intersection(self.core(), *[m.core() for m in mass_functions])
        else:
            focal = self.focal()
            if not focal:
                return frozenset()
            else:
                return frozenset.union(*focal)
    
    def all(self):
        return powerset(self.frame())
    
    def bel(self, hypothesis=None):
        
        if hypothesis is None:
            return {h:self.bel(h) for h in powerset(self.core())}
        else:
            hypothesis = MassFunction._convert(hypothesis)
            if not hypothesis:
                return 0.0
            else:
                return fsum([v for (h, v) in self.items() if h and hypothesis.issuperset(h)])
    
    def pl(self, hypothesis=None):
        if hypothesis is None:
            return {h:self.pl(h) for h in powerset(self.core())}
        else:
            hypothesis = MassFunction._convert(hypothesis)
            if not hypothesis:
                return 0.0
            else:
                return fsum([v for (h, v) in self.items() if hypothesis & h])
    
    def q(self, hypothesis=None):
        if hypothesis is None:
            return {h:self.q(h) for h in powerset(self.core())}
        else:
            if not hypothesis:
                return 1.0
            else:
                return fsum([v for (h, v) in self.items() if h.issuperset(hypothesis)])
    
    def __and__(self, mass_function):
        return self.combine_conjunctive(mass_function)
    
    def __or__(self, mass_function):
        return self.combine_disjunctive(mass_function)
    
    def __str__(self):
        hyp = sorted([(v, h) for (h, v) in self.items()], reverse=True)
        return "{" + "; ".join([str(set(h)) + ":" + str(v) for (v, h) in hyp]) + "}"
    
    def __mul__(self, scalar):
        if not isinstance(scalar, float):
            raise TypeError('Can only multiply by a float value.')
        m = MassFunction()
        for (h, v) in self.items():
            m[h] = v * scalar
        return m
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __add__(self, m):
        if not isinstance(m, MassFunction):
            raise TypeError('Can only add two mass functions.')
        result = self.copy()
        for (h, v) in m.items():
            result[h] += v
        return result
    
    def weight_function(self):
        weights = dict()
        q = self.q()
        theta = self.frame()
        for h in powerset(theta):
            if len(h) < len(theta): # weight is undefined for theta
                sets = [h | c for c in powerset(theta - h)]
                q_even = reduce(mul, [q[h2] for h2 in sets if len(h2) % 2 == 0], 1.0)
                q_odd = reduce(mul, [q[h2] for h2 in sets if len(h2) % 2 == 1], 1.0)
                if len(h) % 2 == 0:
                    weights[h] = q_odd / q_even
                else:
                    weights[h] = q_even / q_odd
        return weights
    
    def combine_conjunctive(self, mass_function, normalization=True, sample_count=None, importance_sampling=False):
        return self._combine(mass_function, rule=lambda s1, s2: s1 & s2, normalization=normalization, sample_count=sample_count, importance_sampling=importance_sampling)
    
    def combine_disjunctive(self, mass_function, sample_count=None):

        return self._combine(mass_function, rule=lambda s1, s2: s1 | s2, normalization=False, sample_count=sample_count, importance_sampling=False)
    
    def combine_cautious(self, mass_function):
        w1 = self.weight_function()
        w2 = mass_function.weight_function()
        w_min = {h:min(w1[h], w2[h]) for h in w1}
        theta = self.frame()
        m = MassFunction({theta:1.0})
        for h, w in w_min.items():
            m_simple = MassFunction({theta:w, h:1.0 - w})
            m = m.combine_conjunctive(m_simple, normalization=False)
        return m
    
    def _combine(self, mass_function, rule, normalization, sample_count, importance_sampling):
        combined = self
        if isinstance(mass_function, MassFunction):
            mass_function = [mass_function] # wrap single mass function
        for m in mass_function:
            if not isinstance(m, MassFunction):
                raise TypeError("expected type MassFunction but got %s; make sure to use keyword arguments for anything other than mass functions" % type(m))
            if sample_count == None:
                combined = combined._combine_deterministic(m, rule)
            else:
                if importance_sampling and normalization:
                    combined = combined._combine_importance_sampling(m, sample_count)
                else:
                    combined = combined._combine_direct_sampling(m, rule, sample_count)
        if normalization:
            return combined.normalize()
        else:
            return combined
    
    def _combine_deterministic(self, mass_function, rule):
        combined = MassFunction()
        for (h1, v1) in self.items():
            for (h2, v2) in mass_function.items():
                combined[rule(h1, h2)] += v1 * v2
        return combined
    
    def _combine_direct_sampling(self, mass_function, rule, sample_count):
        combined = MassFunction()
        samples1 = self.sample(sample_count)
        samples2 = mass_function.sample(sample_count)
        for i in range(sample_count):
            combined[rule(samples1[i], samples2[i])] += 1.0 / sample_count
        return combined
    
    def _combine_importance_sampling(self, mass_function, sample_count):
        combined = MassFunction()
        for (s1, n) in self.sample(sample_count, as_dict=True).items():
            weight = mass_function.pl(s1)
            for s2 in mass_function.condition(s1).sample(n):
                combined[s2] += weight
        return combined
    
    def combine_gbt(self, likelihoods, normalization=True, sample_count=None, importance_sampling=True):
        core = self.core() # restrict to generally compatible likelihoods
        if isinstance(likelihoods, dict):
            likelihoods = list(likelihoods.items())
        likelihoods = [l for l in likelihoods if l[1] > 0 and l[0] in core]
        if sample_count == None: # deterministic
            return self.combine_conjunctive(MassFunction.gbt(likelihoods), normalization=normalization)
        else: # Monte-Carlo
            if not normalization: # only use importance sampling in case of normalization
                importance_sampling = False
            combined = MassFunction()
            for s, n in self.sample(sample_count, as_dict=True).items():
                if importance_sampling:
                    compatible_likelihoods = [l for l in likelihoods if l[0] in s]
                    weight = 1.0 - reduce(mul, [1.0 - l[1] for l in compatible_likelihoods], 1.0)
                else:
                    compatible_likelihoods = likelihoods
                if not compatible_likelihoods:
                    continue
                if normalization:
                    empty_mass = reduce(mul, [1.0 - l[1] for l in compatible_likelihoods], 1.0)
                for _ in range(n):
                    rv = [random() for _ in range(len(compatible_likelihoods))]
                    subtree_mass = 1.0
                    hyp = set()
                    for k in range(len(compatible_likelihoods)):
                        l = compatible_likelihoods[k][1]
                        norm = 1.0 if hyp or not normalization else 1.0 - empty_mass / subtree_mass
                        if l / norm > rv[k]:
                            hyp.add(compatible_likelihoods[k][0])
                        else:
                            subtree_mass *= 1.0 - l   # only relevant for negative case
                    if importance_sampling:
                        combined[hyp] += weight
                    else:
                        combined[hyp & s] += 1.0
            if normalization:
                return combined.normalize()
            else:
                return combined
    
    def condition(self, hypothesis, normalization=True):
      
        m = MassFunction({MassFunction._convert(hypothesis):1.0})
        return self.combine_conjunctive(m, normalization=normalization)
    
    def conflict(self, mass_function, sample_count=None):
        # compute full conjunctive combination (could be more efficient)
        c = self.combine_conjunctive(mass_function, normalization=False, sample_count=sample_count)[frozenset()]
        if c >= 1.0 - 1E-8:
            return float("inf")
        else:
            return -log(1.0 - c)
    
    def normalize(self):
        if frozenset() in self:
            del self[frozenset()]
        mass_sum = fsum(self.values())
        if mass_sum != 1.0:
            for (h, v) in self.items():
                self[h] = v / mass_sum
        return self
    
    def prune(self):
        remove = [h for (h, v) in self.items() if v == 0.0]
        for h in remove:
            del self[h]
        return self
    
    def markov(self, transition_model, sample_count=None):
        updated = MassFunction()
        if sample_count == None: # deterministic
            for k, v in self.items():
                predicted = None
                for e in k:
                    if predicted == None:
                        predicted = transition_model(e)
                    else:
                        predicted |= transition_model(e)
                for kp, vp in predicted.items():
                    updated[kp] += v * vp
        else: # Monte-Carlo
            for s, n in self.sample(sample_count, as_dict=True).items():
                unions = [[] for _ in range(n)]
                for e in s:
                    ts = transition_model(e, n)
                    for i, t in enumerate(ts):
                        unions[i].extend(t)
                for u in unions:
                    updated[u] += 1.0 / sample_count
        return updated
    
    def map(self, function)
        m = MassFunction()
        for (h, v) in self.items():
            m[self._convert(function(h))] += v
        return m
    
    def pignistic(self):
        p = MassFunction()
        for (h, v) in self.items():
            if v > 0.0:
                size = float(len(h))
                for s in h:
                    p[(s,)] += v / size
        return p.normalize()
    
    def local_conflict(self):
        if self[frozenset()] > 0.0:
            return float('nan')
        c = 0.0
        for (h, v) in self.items():
            if v > 0.0:
                c += v * log(len(h) / v, 2)
        return c
    
    def hartley_measure(self):
        return fsum([v * log(len(h), 2) for h, v in self.items()])
    
    def norm(self, m, p=2):
        d = fsum([(v - m[h])**p for (h, v) in self.items()])
        for (h, v) in m.items():
            if h not in self:
                d += v**p
        return d**(1.0 / p)
    
    def is_compatible(self, m):
        return all([self.pl(h) >= v for (h, v) in m.items()])
    
    def sample(self, n, quantization=True, as_dict=False):
        if not isinstance(n, int):
            raise TypeError("n must be int")
        samples = {h:0 for h in self} if as_dict else []
        mass_sum = fsum(self.values())
        if quantization:
            remainders = []
            remaining_sample_count = n
            for (h, v) in self.items():
                fraction = n * v / mass_sum
                quotient = int(fraction)
                if quotient > 0:
                    if as_dict:
                        samples[h] = quotient
                    else:
                        samples.extend([h] * quotient)
                remainders.append((h, fraction - quotient))
                remaining_sample_count -= quotient
            remainders.sort(reverse=True, key=lambda hv: hv[1])
            for h, _ in remainders[:remaining_sample_count]:
                if as_dict:
                    samples[h] += 1
                else:
                    samples.append(h)
        else:
            rv = [uniform(0.0, mass_sum) for _ in range(n)]
            hypotheses = sorted(self.items(), reverse=True, key=lambda hv: hv[1])
            for i in range(n):
                mass = 0.0
                for (h, v) in hypotheses:
                    mass += v
                    if mass >= rv[i]:
                        if as_dict:
                            samples[h] += 1
                        else:
                            samples.append(h)
                        break
        if not as_dict:
            shuffle(samples)
        return samples
    
    def is_probabilistic(self):
        return all([len(h) == 1 for h in self.keys()])
    
    def sample_probability_distributions(self, n):
        samples = [MassFunction() for _ in range(n)]
        for i in range(n):
            for (h, v) in self.items():
                if len(h) == 1:
                    samples[i][h] += v
                else:
                    rv = [random() for _ in range(len(h))]
                    total = fsum(rv)
                    for k, s in enumerate(h):
                        samples[i][{s}] += rv[k] * v / total
        return samples
    
    def max_bel(self):

        return self._max_singleton(self.bel)
    
    def max_pl(self):
        return self._max_singleton(self.pl)
    
    def _max_singleton(self, f):
        st = self.singletons()
        if st:
            value_list = [(f(s), s) for s in st]
            shuffle(value_list)
            return max(value_list)[1]
        else:
            return None
    
    def to_dict(self):
        if not self.is_probabilistic():
            raise Exception('mass function must only contain singletons')
        return {tuple(h)[0]:v for h, v in self.items()}
    
    @staticmethod
    def from_dict(d):
        if isinstance(d, MassFunction):
            return d
        else:
            return MassFunction({frozenset((h,)):v for h, v in d.items()})
    
    @staticmethod
    def from_possibility(poss):
        if isinstance(poss, MassFunction):
            poss = poss.to_dict() # remove enclosing sets
        H, P = zip(*sorted(poss.items(), key=lambda e: e[1], reverse=True)) # sort possibility values in descending order
        m = MassFunction()
        m[H] = P[-1]
        for i in range(len(H) - 1):
            m[H[:i + 1]] = P[i] - P[i + 1]
        return m
    
    @staticmethod
    def pignistic_inverse(p):
        p = MassFunction.from_dict(p)
        poss = MassFunction({h1:fsum([min(p[h1], p[h2]) for h2 in p.keys()]) for h1 in p.keys()})
        return MassFunction.from_possibility(poss)
    
    @staticmethod
    def _to_array_index(hypothesis, frame):
        index = 0
        for i, s in enumerate(frame):
            if s in hypothesis:
                index += 2**i
        return index
    
    @staticmethod
    def _from_array_index(index, frame):
        hypothesis = set()
        for i, s in enumerate(frame):
            if 2**i & index:
                hypothesis.add(s)
        return frozenset(hypothesis)
    
    def to_array(self, frame):
        a = numpy.zeros(2**len(frame))
        for h, v in self.items():
            a[MassFunction._to_array_index(h, frame)] = v
        return a
    
    @staticmethod
    def from_array(a, frame):
        m = MassFunction()
        for i, v in enumerate(a):
            if v > 0.0:
                m[MassFunction._from_array_index(i, frame)] = v
        return m
    
    @staticmethod
    def _confidence_intervals(histogram, alpha):
        p_lower = {}
        p_upper = {}
        a = chi2.ppf(1. - alpha / len(histogram), 1)
        n = float(sum(histogram.values()))
        for h, n_h in histogram.items():
            delta_h = a * (a + 4. * n_h * (n - n_h) / n)
            p_lower[h] = (a + 2. * n_h - sqrt(delta_h)) / (2. * (n + a))
            p_upper[h] = (a + 2. * n_h + sqrt(delta_h)) / (2. * (n + a))
        return p_lower, p_upper
    
    @staticmethod
    def from_samples(histogram, method='idm', alpha=0.05, s=1.0):
        if not isinstance(histogram, dict):
            raise TypeError('histogram must be of type dict')
        for v in histogram.values():
            if not isinstance(v, int):
                raise TypeError('all histogram values must be of type int')
        if not histogram:
            return MassFunction()
        if sum(histogram.values()) == 0: # return vacuous/uniform belief if there are no samples
            vac = MassFunction({tuple(histogram.keys()):1})
            if method == 'bayesian':
                return vac.pignistic()
            else:
                return vac
        if method == 'bayesian':
            return MassFunction({(h,):v + s for h, v in histogram.items()}).normalize()
        elif method == 'idm':
            return MassFunction._from_samples_idm(histogram, s)
        elif method == 'maxbel':
            return MassFunction._from_samples_maxbel(histogram, alpha)
        elif method == 'maxbel-ordered':
            return MassFunction._from_samples_maxbel(histogram, alpha, ordered=True)
        elif method == 'mcd':
            return MassFunction._from_samples_mcd(histogram, alpha)
        elif method == 'mcd-approximate':
            return MassFunction._from_samples_mcd(histogram, alpha, approximate=True)
        raise ValueError('unknown method: %s' % method)
    
    @staticmethod
    def _from_samples_idm(histogram, s):
        total = sum(histogram.values())
        m = MassFunction()
        for h, c in histogram.items():
            m[(h,)] = float(c) / (total + s)
        m[MassFunction._convert(histogram.keys())] = float(s) / (total + s)
        return m
    
    @staticmethod
    def _from_samples_maxbel(histogram, alpha, ordered=False):
        p_lower, p_upper = MassFunction._confidence_intervals(histogram, alpha)
        def p_lower_set(hs):
            l = u = 0
            for h in H:
                if h in hs:
                    l += p_lower[h]
                else:
                    u += p_upper[h]
            return max(l, 1 - u)
        if ordered:
            H = sorted(histogram.keys())
            m = MassFunction()
            for i1, h1 in enumerate(H):
                m[(h1,)] = p_lower[h1]
                for i2, h2 in enumerate(H[i1 + 1:]):
                    i2 += i1 + 1
                    if i2 == i1 + 1:
                        v = p_lower_set(H[i1:i2 + 1]) - p_lower[h1] - p_lower[h2]
                    else:
                        v = p_lower_set(H[i1:i2 + 1]) - p_lower_set(H[i1 + 1:i2 + 1]) - p_lower_set(H[i1:i2]) + p_lower_set(H[i1 + 1:i2])
                    if v > 0:
                        m[H[i1:i2 + 1]] = v
            return m
        else:
            H = list(histogram.keys())
            L = 2**len(H)
            initial = numpy.zeros(L)
            cons = []
            singletons = lambda index: [i for i in range(len(H)) if 2**i & index]
            # constraint (24)
            bel = lambda index, m: fsum(m[sum([2**i for i in h_ind])] for h_ind in powerset(singletons(index)))
            c24 = lambda m, i: p_lower_set(MassFunction._from_array_index(i, H)) - bel(i, m)
            for i in range(L):
                cons.append(partial(c24, i=i))
            # constraint (25)
            cons.append(lambda m: m.sum() - 1.0)
            cons.append(lambda m: 1.0 - m.sum())
            # constraint (26)
            for i in range(L):
                cons.append(partial(lambda m, i_s: m[i_s], i_s=i))
            f = lambda m: -1 * 2**len(H) * fsum([m[i] * 2**(-len(singletons(i))) for i in range(L)])
            m_optimal = fmin_cobyla(f, initial, cons, disp=0)
            return MassFunction.from_array(m_optimal, H)
        
    @staticmethod
    def _from_samples_mcd(histogram, alpha, approximate=False):
        p_lower, p_upper = MassFunction._confidence_intervals(histogram, alpha)
        H = list(histogram.keys())
        if approximate:
            # approximate possibility distribution
            poss = {h1:min(1, fsum([min(p_upper[h1], p_upper[h2]) for h2 in H])) for h1 in H}
        else:
            # optimal possibility distribution (based on linear programming)
            poss = {h:0. for h in H}
            for k, h_k in enumerate(H):
                S_k = {l for l in range(len(H)) if p_lower[H[l]] >= p_upper[h_k]}
                S_k.add(k)
                I_k = {l for l in range(len(H)) if p_upper[H[l]] < p_lower[h_k]}
                P_k = set(range(len(H))).difference(S_k.union(I_k))
                for A in powerset(P_k):
                    G = S_k.union(A)
                    G_c = set(range(len(H))).difference(G)
                    cons = []
                    # constraint (26)
                    for i, h in enumerate(H):
                        cons.append(partial(lambda p, i_s, p_s: p[i_s] - p_s, i_s=i, p_s=p_lower[h])) # lower bound
                        cons.append(partial(lambda p, i_s, p_s: p_s - p[i_s], i_s=i, p_s=p_upper[h])) # upper bound
                    # constraint (27)
                    cons.append(lambda p: 1. - sum(p))
                    cons.append(lambda p: sum(p) - 1.)
                    # constraint (30)
                    for i in G:
                        cons.append(partial(lambda p, i_s: p[i_s] - p[k], i_s=i))
                    # constraint (31)
                    for i in G_c:
                        cons.append(partial(lambda p, i_s: p[k] - p[i_s], i_s=i))
                    initial = [1.0 / len(H)] * len(H)
                    f = lambda p: -(fsum([p[i] for i in G_c]) + len(G) * p[k])
                    poss_optimal = fmin_cobyla(f, initial, cons, disp=0)
                    poss[h_k] = max(poss[h_k], -f(poss_optimal))
        return MassFunction.from_possibility(poss)


def gbt_m(hypothesis, likelihoods, normalization=True):

    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    q = gbt_q(hypothesis, likelihoods, normalization)
    return q * reduce(mul, [1.0 - l[1] for l in likelihoods if l[0] not in hypothesis], 1.0)

def gbt_bel(hypothesis, likelihoods, normalization=True):
    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    eta = _gbt_normalization(likelihoods) if normalization else 1.0
    exc = reduce(mul, [1.0 - l[1] for l in likelihoods if l[0] not in hypothesis], 1.0)
    all_hyp = reduce(mul, [1.0 - l[1] for l in likelihoods], 1.0)
    return eta * (exc - all_hyp)

def gbt_pl(hypothesis, likelihoods, normalization=True):

    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    eta = _gbt_normalization(likelihoods) if normalization else 1.0
    return eta * (1.0 - reduce(mul, [1.0 - l[1] for l in likelihoods if l[0] in hypothesis], 1.0))
    
def gbt_q(hypothesis, likelihoods, normalization=True):

    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    eta = _gbt_normalization(likelihoods) if normalization else 1.0
    return eta * reduce(mul, [l[1] for l in likelihoods if l[0] in hypothesis], 1.0)

def gbt_pignistic(singleton, likelihoods):

    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    singleton_lh = None
    lh_values = []
    for h, v in likelihoods:
        if h == singleton:
            singleton_lh = v
        else:
            lh_values.append(v)
    if singleton_lh is None:
        raise ValueError('singleton %s is not contained in likelihoods' % repr(singleton))
    m_sum = _gbt_pignistic_recursive(lh_values, 0)
    eta = _gbt_normalization(likelihoods)
    return sum([eta * v * singleton_lh / (c + 1.) for c, v in enumerate(m_sum)])

def _gbt_pignistic_recursive(likelihoods, i):

    if i == len(likelihoods) - 1:
        m_sum = [0.] * (len(likelihoods) + 1)
        m_sum[0] = 1. - likelihoods[i]
        m_sum[1] = likelihoods[i]
        return m_sum
    else:
        m_sum = _gbt_pignistic_recursive(likelihoods, i + 1)
        m_sum_inc = [0.] * (len(likelihoods) + 1)
        m_sum_exc = [0.] * (len(likelihoods) + 1)
        for k in range(len(likelihoods) + 1):
            if k < len(likelihoods):
                m_sum_inc[k+1] = m_sum[k] * likelihoods[i]
            m_sum_exc[k] = m_sum[k] * (1. - likelihoods[i])
            m_sum[k] = m_sum_inc[k] + m_sum_exc[k]
        return m_sum

def _gbt_normalization(likelihoods):
    return 1.0 / (1.0 - reduce(mul, [1.0 - l[1] for l in likelihoods], 1.0))
