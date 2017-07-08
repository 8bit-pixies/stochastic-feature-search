import numpy as np
import pandas as pd

from scipy.stats import poisson, gamma

from itertools import chain

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import make_scorer

# for sampling
import random # use random.choice? and random.sample for interactions

import itertools
from itertools import combinations

def create_model(additional_feats=[]):
    pipeline = additional_feats[:]
    pipeline.append(('SGD_regressor', SGDRegressor(loss='squared_loss', penalty='elasticnet')))
    model = Pipeline(pipeline[:])
    return model

def eval_pipeline(additional_feats, X, y, verbose=True):
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
def output_prob_state(k, l=None, c=0.4, gamma_loc=10, gamma_scale=11, show_lambda=False):
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
    
    if l is None:
        # generate a sample l from Gamma
        l = gamma.rvs(gamma_loc+k, gamma_scale,size=1)[0]
    if show_lambda:
        print("Lambda selected to be: {}".format(l))
    
    if k <= 1:
        return 1, 0, 0

    
    poisson_obj = poisson(l)
    birth = c*min(1, (poisson_obj.pmf(k+1)/poisson_obj.pmf(k)))
    death = c*min(1, (poisson_obj.pmf(k)/poisson_obj.pmf(k+1)))
    change = 1.0-birth-death
    
    # output the probabilities...which are used for the generated unichr
    # slot.

    return birth, death, change

def output_action(u, birth, death, change):
    if birth+death+change <= 0.9999 and birth+death+change >= 1.00001:
        raise Exception("birth, death, change does not appear to be a probability")
    if u <= birth:
        return 'birth'
    if u <= birth + death:
        return 'death'
    else:
        return 'change'

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
    def __init__(self, indices, knots, signs):
        #self.indices = np.array(indices)
        #self.knots = np.array(knots)
        #self.signs = np.array(signs)
        self.indices = indices
        self.knots   = knots  
        self.signs   = signs  
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_subset = np.array(X[self.indices])
        else:
            X_subset = X[:, self.indices]
        for idx, knot in enumerate(self.knots):
            X_subset[:, idx] = np.maximum(X_subset[:, idx]-knot, 0) * self.signs[idx]
        
        # if multiple collapse by interaction        
        return np.prod(X_subset, axis=1).reshape(-1, 1)

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
    
    def construct_pipeline(self, colnames=True):
        model_matrix = [('base model', BaseModel())]
        col_names = []
        for basis, params in zip(self.basis, self.params):
            model_name = "B_{}".format("".join(str(x) for x in list(basis)))
            model_obj  = Hinge(np.array(basis), np.array(params['knots']), np.array(params['signs']))
            col_names.append(model_name)
            model_matrix.append((model_name, model_obj))
        if colnames:
            return [('union', FeatureUnion(model_matrix))], col_names[:]
        else:
            return [('union', FeatureUnion(model_matrix))]
    
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
        idx_pop = [idx for idx, set_b in enumerate(basis_set) if set(basis) == set_b][0]
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
    
    def perform_action(self, mode='birth'):
        """
        mode is one of 'birth', 'death', 'change'
        
        if selected will return the basis of interest. 
        """
        basis_set = self._get_basis_set()        
                
        if mode == 'birth':
            try:
                s = self.X.columns
            except:
                s = list(range(self.X.shape[1]))
            # s = list(range(X.shape[1]))
            max_size = self.interaction+1
            all_combin = list(chain.from_iterable(set(list(combinations(s, r))) for r in range(1, max_size)))
            
            # now based on this go ahead and...do stuff!
            basis_set = self._get_basis_set()
            valid_basis = [x for x in all_combin if x not in basis_set]
            return random.choice(valid_basis)
        elif mode in ['death', 'change']:
            return random.choice(basis_set)
        else:
            raise Exception("mode: {} not valid in perform_action".format(mode))


# last step is to calculate the acceptance criteria..
def bmars_sample_basis(X, basis, params=None, mode='dict'):
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
    if isinstance(X, pd.DataFrame):
        X_subset = np.array(X[basis])
    else:
        X_subset = X[:, basis]
    
    # redrawing signs is easy...it is random choice of -1, 1
    import random
    signs = [random.choice([-1, 1]) for _ in basis]
    
    knots = np.apply_along_axis(np.random.choice, 0, X_subset)
    
    # create new param set
    new_param = {}
    new_param['sign'] = signs
    new_param['knot'] = knots
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
    
    bayes_ratio = accept_bayes_factor(X, y, current_BMARS, proposed_BMARS, mode)
    prior_ratio = accept_prior_ratio(X, y, l, interaction, current_BMARS, proposed_BMARS, mode)
    proposal_ratio = accept_proposal_ratio(X, y, l, interaction, current_BMARS, proposed_BMARS, mode)
    
    alpha = min(1.0, bayes_ratio * prior_ratio * proposal_ratio)
    return alpha


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
    # we will calculate the likelihood based on the pipeline...
    # for gaussian it is straight forward...
    # create model...
    
    # likelihood ratio...
    # you need to "integrate" out all possible hyper parameters to get the bayes factor here...    
    # if mode is change - we will probably want to use a point estimate. of the two models
    # but we will leave this alone for now.
    if mode == 'change':
        current_model = create_model(current_BMARS.construct_pipeline(False))
        current_model.fit(X, y)
        
        proposed_model = create_model(proposed_BMARS.construct_pipeline(False))
        proposed_model.fit(X, y)
        
        y_hat_current = current_model.predict(X)
        y_hat_proposed = proposed_model.predict(X)
        bayes_factor = gaussian_likelihood(y, y_hat_proposed)/gaussian_likelihood(y, y_hat_current)    
    elif mode == 'birth':
        model = proposed_BMARS.export()
        current_basis = current_BMARS.export()['basis']
        propose_basis = proposed_BMARS.export()['basis']
        current_model = create_model(current_BMARS.construct_pipeline(False))
        current_model.fit(X, y)
        
        # find the new basis...
        new_basis = [x for x in propose_basis if set(x) not in [set(y) for y in current_basis]][0]
        
        # we know the likelihood for the current one? so only need to iterate over the new basis...
        y_hat_current = current_model.predict(X)
        current_likelihood = gaussian_likelihood(y, y_hat_current)   
        
        # need to perform some MC for proposed likelihood.
        # generate all combinations...
        # faster way might be to get histogram of values.
        pos_knots = []
        
        for b_col in new_basis:
            # if there are duplicates it is "fine"
            # as it will be reflective of the duplication of 
            # knot points
            knots = np.array(X[:, b_col]).tolist()
            knot_p = [knots, [-1, 1]]
            pos_knot_combo = list(itertools.product(*knot_p))
            pos_knots.append(pos_knot_combo)
        
        # it will be knot points, with the last param being sign.   
        # this will be list of tuples of tuple..
        
        # [((knot, sign), ... # basis)]
        pos_comb_knots = list(itertools.product(*pos_knots))
        
        # sample how many? - say 30
        n_sample = min(30, len(pos_comb_knots))
        import random
        eval_points = random.sample(pos_comb_knots, n_sample) 
        
        proposed_params_base = current_BMARS.export()
        def add_param(new_basis, comb_knots):   
            # def add_basis(self, basis, knot, sign):
            knots = [x[0] for x in list(comb_knots)]
            signs = [x[1] for x in list(comb_knots)]
            return {'basis': new_basis, 'knot': knots, 'sign': signs}
        
        propose_likelihoods = []
        for basis_knot in eval_points:
            proposed_model = BMARS(**model)
            params = add_param(new_basis, basis_knot)
            proposed_model.add_basis(**params)
            proposed_model_fitted = create_model(proposed_model.construct_pipeline(False)).fit(X, y)
            y_hat_propose = proposed_model_fitted.predict(X)
            propose_likelihoods.append(gaussian_likelihood(y, y_hat_propose))
        propose_likelihood = np.mean(propose_likelihoods)
        bayes_factor = current_likelihood/propose_likelihood
    elif mode == 'death':
        return accept_bayes_factor(X, y, proposed_BMARS, current_BMARS, mode='birth')
    else:
        # do exhaustive search - or use percnetiles for histogram information for faster 
        # eval in MC sense.
        raise Exception("Invalid mode: {} chosen. Please choose mode in 'birth', 'death', 'change' in accept_bayes_factor".format(mode))
        
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
    try:
        s = X.columns
    except:
        s = list(range(X.shape[1]))
    max_size = interaction+1
    all_combin = list(chain.from_iterable(set(list(combinations(s, r))) for r in range(1, max_size)))    
    basis_set = current_BMARS._get_basis_set()
    valid_basis = [x for x in all_combin if x not in basis_set]
    N = len(valid_basis)
    n = X.shape[0]
    
    current_param  = current_BMARS.export()
    proposed_param = proposed_BMARS.export()
    current_basis = current_param['basis']
    propose_basis = proposed_param['basis']
    k = len(current_basis)
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
    n = X.shape[0]
    
    if mode == 'birth':
        # propose death / propose birth
        try:
            s = X.columns
        except:
            s = list(range(X.shape[1]))
        # s = list(range(X.shape[1]))
        max_size = current_BMARS.interaction+1
        all_combin = list(chain.from_iterable(set(list(combinations(s, r))) for r in range(1, max_size)))
        
        # now based on this go ahead and...do stuff!
        basis_set = current_BMARS._get_basis_set()
        valid_basis = [x for x in all_combin if x not in basis_set]
        N = len(valid_basis)        
        
        new_basis = [set(x) for x in propose_basis if set(x) not in [set(y) for y in current_basis]][0]
        J_k1 = len(new_basis)
        b_k1, d_k1, p_k1 = output_prob_state(k+1, l)
        d_proposal = 1 if k == 0 else (d_k1/k)
        b_proposal = (b_k/(N*((2*n)**J_k1)))
        return d_proposal / b_proposal
    if mode == 'death':
        return 1/accept_proposal_ratio(X, y, l, interaction, proposed_BMARS, current_BMARS, mode='birth')
        
    raise Exception("mode: {} not one of 'birth', 'death', 'change' in accept_proposal_ratio.")
        
    
    
    
