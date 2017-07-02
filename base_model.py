import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

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
class BMARS(object):
    def __init__(self, X, interactin=2):
        # add other params later...
        self.X = X # X is your base matrix, i.e. when Basis = 1
        self.interaction = interaction
        self.basis = []
        self.params = {} # dict of list with parameters related to basis
    
    def build_pipeline(self):
        # returns pipeline object...
        pass
        
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
        s = X.columns
        # s = list(range(X.shape[1]))
        max_size = self.interaction+1
        all_combin = chain.from_iterable(set(list(combinations(s, r))) for r in range(max_size))
        
        # now based on this go ahead and...do stuff!
        valid_basis = [x for x in all_combin if x not in self.basis]
        return valid_basis
        
        
        
        
        
        



# last step is to calculate the acceptance criteria..
def acceptance():
    # this probably should be a class? maybe
    # this is...
    # min(1, bayes_factor x prior_ratio x proposal_ratio x jacobian {1})
    pass

# bayes factor
def accept_bayes_factor(X, proposal_model, current_model):
    """
    Do something and just us MC to approximate for now...
    ask Richard what im doing with this part...
    """
    pass
    
    
    
