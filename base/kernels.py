from baybe.surrogates.gaussian_process.kernel_factory import KernelFactory
# BayBE kernels (NOT GPyTorch!)
# BE CAREFUL !!
# Kernels share names in gpytorch and baybe BUT are not the same!!
from baybe.kernels import ScaleKernel, MaternKernel, RBFKernel
from baybe.priors.basic import GammaPrior, LogNormalPrior
# from gpytorch.priors import UniformPrior
import math
import numpy as np


class MaternKernelFactory(KernelFactory):
    """
    Normalize each parameter group to have similar influence.
    Simpler than full block ARD, but effective.
    """

    def __init__(self, prior_set="max_custom_0", n_dim=None, kernel_name_user='Matern'):
        self.prior_set = prior_set
        self.n_dim = n_dim
        self.kernel_name_user = kernel_name_user
    
    def __call__(self, searchspace, train_x, train_y):

        if self.prior_set == "BayBE8D":
            # BayBE 8D ini   
            lengthscale_prior = GammaPrior(1.2, 1.1)
            lengthscale_initial_value = 0.2
            outputscale_prior = GammaPrior(5.0, 0.5)
            outputscale_initial_value = 8.0

        elif self.prior_set == "BayBE75D":
            # BayBE 75D ini
            lengthscale_prior = GammaPrior(concentration=2.5, rate=0.55)
            lengthscale_initial_value = 6.0
            outputscale_prior = GammaPrior(concentration=3.5, rate=0.15)
            outputscale_initial_value = 15.0    

        elif self.prior_set == "EDBO+" or self.prior_set == "EDBO_QM":
            # EDBO+ ini == QM features prior from EDBO
            lengthscale_prior = GammaPrior(concentration=2.0, rate=0.2)
            lengthscale_initial_value = 5.0
            outputscale_prior = GammaPrior(concentration=5.0, rate=0.5)
            outputscale_initial_value = 8.0                

        # EDBO
        #   References:
        # * https://github.com/b-shields/edbo/blob/master/edbo/bro.py#L664
        # * https://doi.org/10.1038/s41586-021-03213-y
        # * https://emdgroup.github.io/baybe/stable/_modules/baybe/surrogates/gaussian_process/presets/edbo.html#EDBOKernelFactory

        elif self.prior_set == "EDBO_MORDRED":
            lengthscale_prior = GammaPrior(concentration=2.0, rate=0.1)
            lengthscale_initial_value = 10.0
            outputscale_prior = GammaPrior(concentration=2.0, rate=0.1)
            outputscale_initial_value = 10.0  

        elif self.prior_set == "EDBO_OHE":
            lengthscale_prior = GammaPrior(concentration=3.0, rate=1.0)
            lengthscale_initial_value = 2.0
            outputscale_prior = GammaPrior(concentration=5.0, rate=0.2)
            outputscale_initial_value = 20.0    

        elif self.prior_set == "max_custom_0":
            lengthscale_prior = GammaPrior(20.0, 0.5)
            lengthscale_initial_value = 10.0
            outputscale_prior = GammaPrior(2.0, 0.1)
            outputscale_initial_value = 10.0          

        elif self.prior_set == "max_LHS_0":
            lengthscale_prior = GammaPrior(20.0, 0.3)
            lengthscale_initial_value = 20.0
            outputscale_prior = GammaPrior(2.0, 0.1)
            outputscale_initial_value = 10.0 
        
        elif self.prior_set == "BayBE_adaptive":
            # BayBE default factory
            if self.n_dim is None:
                raise ValueError("n_dim must be given for adaptive prior!")
            
            _DIM_LIMITS = (8, 75)
            lengthscale_prior = GammaPrior(
                np.interp(self.n_dim, _DIM_LIMITS, [1.2, 2.5]),
                np.interp(self.n_dim, _DIM_LIMITS, [1.1, 0.55]),
            )
            lengthscale_initial_value = np.interp(self.n_dim, _DIM_LIMITS, [0.2, 6.0])
            outputscale_prior = GammaPrior(
                np.interp(self.n_dim, _DIM_LIMITS, [5.0, 3.5]),
                np.interp(self.n_dim, _DIM_LIMITS, [0.5, 0.15]),
            )
            outputscale_initial_value = np.interp(self.n_dim, _DIM_LIMITS, [8.0, 15.0])
            
        elif self.prior_set == "LogNormal_DSP":
            # from https://github.com/hvarfner/vanilla_bo_in_highdim
            if self.n_dim is None:
                raise ValueError("n_dim must be given for adaptive prior!")
            
            ls_params = {'loc': 1.4, 'scale': 1.73205}
            ls_params['loc'] = ls_params['loc'] + math.log(self.n_dim) * 0.5
            ls_params['scale'] = (ls_params['scale'] ** 2 + math.log(self.n_dim) * 0.0) **0.5 
            
            lengthscale_prior = LogNormalPrior(**ls_params)
            lengthscale_initial_value = None
            outputscale_prior = GammaPrior(2, 0.15)
            outputscale_initial_value = None
            
        elif self.prior_set == "SBO":
            # from https://github.com/XZT008/Standard-GP-is-all-you-need-for-HDBO
            if self.n_dim is None:
                raise ValueError("n_dim must be given for adaptive prior!")
            # This causes Optimization Failure sometimes. 
            lengthscale_prior = GammaPrior(3.0, 6.0) # Uniform(1e-10, 30) is not supported in BayBE
            lengthscale_initial_value = 1.0 * math.sqrt(self.n_dim)
            outputscale_prior = GammaPrior(2, 0.15)
            outputscale_initial_value = None
            
        elif self.prior_set == 'adaptive_emilien':
            if self.n_dim is None:
                raise ValueError("n_dim must be given for adaptive prior!")
            
            x = math.sqrt(self.n_dim)
            l_mean = 0.4 * x + 4.0 # decided by fitting the result points.
            
            lengthscale_prior = GammaPrior(2.0*l_mean, 2.0)
            lengthscale_initial_value = l_mean
            outputscale_prior = GammaPrior(1.0*l_mean, 1.0) # can use a smaller rate for larger variance.
            outputscale_initial_value = l_mean
        
        # NOTE
        # In Gammaprior(conc, rate), mean = conc/rate, var = conc/rate^2
        # Outputscale should not vary with lengthscale/dimension... But when dim increases, there's more scarcity, then larger outputscale may be needed to explain the increasing uncertainty. But since we are not sure about this "law", we use a small rate in Gammaprior to enable large variance.


        return ScaleKernel(
            MaternKernel(
                nu=2.5,
                lengthscale_prior=lengthscale_prior,
                lengthscale_initial_value=lengthscale_initial_value,
            ) if self.kernel_name_user in ['Matern', 'matern'] else RBFKernel(lengthscale_prior=lengthscale_prior, lengthscale_initial_value=lengthscale_initial_value),
            outputscale_prior=outputscale_prior,
            outputscale_initial_value=outputscale_initial_value,
        )