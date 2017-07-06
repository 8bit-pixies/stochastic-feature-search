import numpy as np
import pandas as pd

from itertools import chain

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion

# for sampling
import random # use random.choice? and random.sample for interactions

def create_model(additional_feats=[]):
    pipeline = additional_feats[:]
    pipeline.append(('SGD_regressor', SGDRegressor(loss='squared_loss', penalty='elasticnet')))
    model = Pipeline(pipeline[:])
    return model

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
        basis_set = self._get_basis_set()
        idx_pop = [idx for idx, set_b in basis_set if set(basis) == set_b][0]
        self.basis.pop(idx_pop)
        self.params.pop(idx_pop)
    
    def _get_params(self, basis):
        # based on a basis set, get the associated parameters...
        # assumes the basis exists
        basis_set = self._get_basis_set()
        idx = [idx for idx, set_b in basis_set if set(basis) == set_b][0]
        return self.params[idx]
    
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
def bmars_sample_basis(X, basis, params, mode='dict'):
    """
    -  X is training data
    -  basis is the columns to be selected for the basis
    -  params is the parameters associated with this basis
    -  mode is one of dict or list, if it is list will return: 
        basis, knots, sign
    
    This function provides another set of parameters for
    *  sign(s)
    *  basis
    
    of chosen one.
    knots and signs are assumed to be uniform.
    
    This function can be used for adding or changing a selected basis
    as switches are assumed to be uniform and independent. 
    """
    X_subset = np.array(X[basis])
    
    # redrawing signs is easy...it is random choice of -1, 1
    import random
    signs = params['signs'][:]
    signs = [random.choice([-1, 1]) for _ in signs]    
    
    knots = np.apply_along_axis(np.random.choice, 0, X_subset)    
    
    # create new param set
    new_param = {}
    new_param['signs'] = signs
    new_param['knots'] = knots
    new_param['basis'] = basis
    
    if mode == 'dict':
        return new_param
    elif mode == 'list':
        return new_param['basis'], new_param['knots'], new_param['signs']
    else:
        raise Exception("Invalid choice of output, mode should be one of 'dict' or 'list'.")

def acceptance_proba(X, y, l, interaction, current_BMARS, proposed_BMARS, mode='change'):
    """
    X is our data
    y is out labels
    l is lambda, the hyperparameter for poisson distribution, where p(k) ~ Poisson(l)
    interaction is max level of interaction
    ... the rest follows.     
    """
    alpha = min(1.0, accept_bayes_factor(X, y, current_BMARS, proposed_BMARS, mode) * accept_prior_ratio(X, y, l, interaction, current_BMARS, proposed_BMARS, mode) * (X, y, l, interaction, current_BMARS, proposed_BMARS, mode))
    return alpha

# bayes factor
def accept_bayes_factor(X, y, current_BMARS, proposed_BMARS, mode='change'):
    """
    Do something and just us MC to approximate for now...
    ask Richard what im doing with this part...
    
    param is empty if it is death - otherwise can provide benefit?    
    basis is the one to: add, remove, change    
    mode is one of "birth", "death", "change"
    """   
    """
    # if it is change we will use a point likelihood
    if mode == 'change':
        return 1.0
    """
    def gaussian_likelihood(y, y_hat):
        """
        assume gaussian iid noise
        """
        y = y.astype(float)
        y_hat = y_hat.astype(float)
        if np.array_equal(y, y_hat):
            return float("-inf")
        l2 = (y-y_hat)**2    
        sigma2 = np.mean(l2)
        n = len(y)
        
        constant = 1.0/(2*np.pi*sigma2)
        
        return (constant ** (n/2) )* np.exp(-constant*np.sum(l2))
    # we will calculate the likelihood based on the pipeline...
    # for gaussian it is straight forward...
    # create model...
    
    # likelihood ratio...
    # you need to "integrate" out all possible hyper parameters to get the bayes factor here...    
    # if mode is change - we will probably want to use a point estimate. of the two models
    # but we will leave this alone for now.
    if mode == 'change':
        current_model = create_model(current_BMARS.construct_pipeline(False))
        current_model.fit(X)
        
        proposed_model = create_model(proposed_BMARS.construct_pipeline(False))
        proposed_model.fit(X)
        
        y_hat_current = current_model.predict(X)
        y_hat_proposed = proposed_model.predict(X)
        bayes_factor = gaussian_likelihood(y, y_hat_proposed)/gaussian_likelihood(y, y_hat_current)    
    else:
        # do exhaustive search - or use percnetiles for histogram information for faster 
        # eval in MC sense.
        pass
        bayes_factor = gaussian_likelihood(y, y_hat_proposed)/gaussian_likelihood(y, y_hat_current)    
    return bayes_factor

def accept_prior_ratio(X, y, l, interaction, current_BMARS, proposed_BMARS, mode='change'):
    """
    l is lambda which is needed for p(k)
    
    """
    if mode == 'change':
        return 1.0
    
    #p(k) * (k!/N^k) * 
    # use soemthing liek this: all_combin = chain.from_iterable(set(list(combinations(s, r))) for r in range(max_size))
    # all knot positions and sign can be simulated....
    
    """p_k
    
    poisson_obj = poisson(l)
    p_k = poisson_obj.pmf(k)
    """
    
    """prior-basis-function
    max_size = interaction+1
    N = chain.from_iterable(set(list(combinations(s, r))) for r in range(max_size))
    
    return np.math.factorial(k)/(N**k)    
    """
    
    """prior for all knot and signs...
    interaction = interaction
    n = X.shape[0]
    (1.0/(2*n))**(sum(range(interaction+1))-1)
    """
    poisson_obj = poisson(l)
    #p_k = poisson_obj.pmf(k)  
    max_size = interaction+1
    N = chain.from_iterable(set(list(combinations(s, r))) for r in range(max_size))    
    n = X.shape[0]
    
    current_param  = current_BMARS.export()
    proposed_param = proposed_BMARS.export()
    current_basis = current_param['basis']
    propose_basis = proposed_param['basis']
    if mode == 'birth':
        # get the basis for the proposed birth..
        new_basis = [set(x) for x in propose_basis if set(x) not in [set(y) for y in current_basis]][0]
        prior_num_basis_ratio = poisson_obj.pmf(k+1)/poisson_obj.pmf(k)
        prior_type_basis_ratio = k/N
        prior_params_ratio = (1.0/(2*n))**(len(new_basis))
        
    elif mode == 'death':
        rm_basis = [set(x) for x in current_basis if set(x) not in [set(y) for y in propose_basis]][0]
        prior_num_basis_ratio = poisson_obj.pmf(k-1)/poisson_obj.pmf(k)
        prior_type_basis_ratio = N/(k-1)
        prior_params_ratio = (1.0/(2*n))**(-len(rm_basis))
    else:
        raise Exception("mode: {} not one of 'birth', 'death', 'change' in accept_prior_ratio.")
    
    return prior_num_basis_ratio * prior_type_basis_ratio * prior_params_ratio

def accept_proposal_ratio(X, y, l, interaction, current_BMARS, proposed_BMARS, mode='change'):
    """
    l is lambda which is needed for p(k)
    
    """
    if mode == 'change':
        return 1.0
    
    #output_prob_state(k, l=None, c=0.4, gamma_loc=10, gamma_scale=11):
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
    current_param  = current_BMARS.export()
    proposed_param = proposed_BMARS.export()
    current_basis = current_param['basis']
    propose_basis = proposed_param['basis']
    
    k = len(current_param['basis'])
    b_k, d_k, p_k = output_prob_state(k, l)
    
    max_size = interaction+1
    N = chain.from_iterable(set(list(combinations(s, r))) for r in range(max_size))    
    n = X.shape[0]
    
    if mode == 'birth':
        # propose death / propose birth
        new_basis = [set(x) for x in propose_basis if set(x) not in [set(y) for y in current_basis]][0]
        J_k1 = len(new_basis)
        b_k1, d_k1, p_k1 = output_prob_state(k+1, l)
        return (d_k1/k)/(b_k/(N*((2*n)**J_k1)))
    if mode == 'death':
        rm_basis = [set(x) for x in current_basis if set(x) not in [set(y) for y in propose_basis]][0]
        
    raise Exception("mode: {} not one of 'birth', 'death', 'change' in accept_proposal_ratio.")
        
    
    
    
