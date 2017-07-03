import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion

# for sampling
import random # use random.choice? and random.sample for interactions

def eval_pipeline(additional_feats=[], X=X_df, y=y, verbose=True):
    #print(additional_feats)
    
    pipeline = additional_feats[:]
    pipeline.append(('SGD_regressor', SGDRegressor(loss='squared_loss', penalty='elasticnet')))
    model = Pipeline(pipeline[:])

    # split data into 10 folds
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(model, X, y, cv=kfold)
    if verbose:
        print("Result: {}".format(results.mean()))
    return results.mean()

# 
def output_prob_state(k, l=None, c=0.4, gamma_loc=10, gamma_scale=11):
    """
    k is the number of basis/transformations in the current state
    l is the lambda parameter for poisson
    c is constant taken to be 0.4
    
    bk = c min (1, p(k+1)/p(k))
    dk = c min (1, p(k)/p(k+1))
    pk = 1-bk-dk
    
    If we have 1 or 0 basis function always return birth prob to be 1. 
    
    This is for calculating:
    
    `b_k`, `d_k`, `p_k` respectively
    """
    if k <= 1:
        return 1, 0, 0
    from scipy.stats import poisson, gamma
    if l is None:
        # generate a sample l from Gamma
        l = gamma.rvs(gamma_loc+k, gamma_scale,size=1)[0]
    
    poisson_obj = poisson(l)
    birth = c*min(1, (poisson_obj.pmf(k+1)/poisson_obj.pmf(k)))
    death = c*min(1, (poisson_obj.pmf(k)/poisson_obj.pmf(k+1)))
    change = 1.0-birth-death
    
    # output the probabilities...which are used for the generated unichr
    # slot.
    return birth, death, change


class BaseModel(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    
    mask : the column indices you wish to keep    
    """
    
    def __init__(self):
        pass
    
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        # converts pandas to numpy array
        return np.array(x)

class Hinge(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    
    x: indices of interest
    t: hinges 
    s: sign
    
    always return 1d vector
    """
    def __init__(self, x, t, s):
        self.indices = list(x)
        self.knots = t
        self.signs = s
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_subset = np.array(X[[self.indices]])        
        for idx, knot in enumerate(self.knots):
            X_subset[:, idx] = np.maximum(X_subset[:, idx]-knot, 0) * self.signs[idx]
        
        # if multiple collapse by interaction        
        return np.prod(X_subset, axis=1)

class BMARS(object):
    def __init__(self, X, interaction=2, basis=[], params=[]):
        # add other params later...
        self.X = X # X is your base matrix, i.e. when Basis = 1
        self.interaction = interaction
        self.basis = basis # this is a list of list, as order matters
        self.params = params # list of dicts with params by index
    
    def export(self):
        bmars = {}
        bmars['X'] = self.X.copy()
        bmars['interaction'] = self.interaction
        bmars['basis'] = self.basis[:]
        bmars['params'] = self.params[:]
        return bmars
    
    def all_moves(self):
        # determines all possible moves
        # in strict MARS, you can only select basis once, and can't be nested
        """
        selected basis will be a dictionary in the form:
        
        *  basis index: list of lists...[[1], [0], [0, 1]] etc...
        *  sign: list (-1, +1)
        *  knots: list (float)
        
        return list of basis which have not yet been chosen...
        """
        # get all possible moves...
        # if X is pandas...
        s = self.X.columns
        # s = list(range(X.shape[1]))
        max_size = self.interaction+1
        all_combin = chain.from_iterable(set(list(combinations(s, r))) for r in range(max_size))
        
        # now based on this go ahead and...do stuff!
        basis_set = self._get_basis_set()
        valid_basis = [x for x in all_combin if x not in basis_set]
        return valid_basis
    
    def construct_pipeline(self, colnames=True):
        model_matrix = [('base model', BaseModel())]
        col_names = []
        for basis, params in zip(self.basis, self.params):
            model_name = "B_{}".format("".join(str(x) for x in list(basis)))
            model_obj  = Hinge(basis, params['knots'], params['signs'])
            col_names.append(model_name)
            model_matrix.append((model_name, model_obj))
        if colnames:
            return FeatureUnion(model_matrix), col_names[:]
        else:
            return FeatureUnion(model_matrix)
    
    def _get_basis_set(self):
        return [set(x) for x in self.basis]
    
    def _add_basis(self, basis, knot, sign):
        """
        Do not use this method directly.
        """
        self.basis.append(basis[:])        
        param = {}
        param['knots'] = knot
        param['signs'] = sign
        self.params.append(param.copy())
    
    def _remove_basis(self, basis):
        idx_pop = [idx for idx, set_b in basis_set if set(basis) == set_b][0]
        self.basis.pop(idx_pop)
        self.params.pop(idx_pop)
    
    def change_basis(self, basis, knot, sign):
        basis_set = self._get_basis_set()
        if not set(basis) in basis_set:
            raise Exception("Cannot find basis {} in current model".format(' '.join(basis)))
        
        # continue
        self._remove_basis(basis)        
        self._add_basis(basis, knot, sign)
    
    def add_basis(self, basis, knot, sign):
        """
        basis is a list as order matters, 
        knot is is a list
        sign is a list of -1, 1
        
        if a basis set already exists, we will replace it...
        """
        # check if it exists...
        basis_set = self._get_basis_set()
        if set(basis) in basis_set:
            self.change_basis(basis, knot, sign)
        else:
            self._add_basis(basis, knot, sign)
    
    def remove_basis(self, basis, knot=None, sign=None):
        """
        remove basis object
        """
        basis_set = self._get_basis_set()
        if not set(basis) in basis_set:
            raise Exception("Cannot find basis {} in current model".format(' '.join(basis)))
        self._remove_basis(basis)  
            
# last step is to calculate the acceptance criteria..
def acceptance(BMARS_obj):
    # this probably should be a class? maybe
    # this is...
    # min(1, bayes_factor x prior_ratio x proposal_ratio x jacobian {1})
    pass

# bayes factor
def accept_bayes_factor(X, BMARS_obj, basis, param={}, mode="change"):
    """
    Do something and just us MC to approximate for now...
    ask Richard what im doing with this part...
    
    param is empty if it is death - otherwise can provide benefit?    
    basis is the one to: add, remove, change    
    mode is one of "birth", "death", "change"
    """
    curr_obj = BMARS(**BMARS_obj.export())
    
    
    pass
    
    
    
