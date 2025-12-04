import pandas as pd
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter, CustomDiscreteParameter, CategoricalParameter
from baybe.objectives import SingleTargetObjective
from baybe.targets import NumericalTarget
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe import Campaign
from baybe.recommenders import BotorchRecommender, RandomRecommender, TwoPhaseMetaRecommender

from base.kernels import MaternKernelFactory
from base.pretrained_repr import PretrainedWrapper, ChemBERTa_Fingerprint, CheMeleonFingerprint, LLM_Fingerprint, Reaction_Fingerprint
from base.utils import custom_fingerprinter, custom_PCA_fingerprinter, custom_PCA_from_substance

from future.LHS import ChunkingRecommender, PCALHS2Recommender, LHS1Recommender, LHS2Recommender
    
def _normalize(pd_col):
    '''
    between 0 and 1.
    '''
    pd_col = pd_col - pd_col.min()
    pd_col = pd_col / ( pd_col.max() - pd_col.min()  )
    return pd_col

def _normalize_100(pd_col):
    '''
    Normalize values between 0 and 100: used to transform "yield-like" targets.
    '''
    pd_col = pd_col - pd_col.min()
    pd_col = pd_col / ( pd_col.max() - pd_col.min()  )
    pd_col = pd_col * 100
    return pd_col

def load_data(dataset):
    """
    """
    dataset = dataset.lower()
    
    FILEPATH_DICT = {
        'shields': 'datasets/shields_dataset.xlsx',
        'buchwald_hartwig': 'datasets/buchwald_hartwig_Dreher_and_Doyle_input_data.xlsx',
        'suzuki': 'datasets/suzuki_shields.csv',
        'ni_catalyzed_1': 'datasets/additive_rxn_screening_plate_1.csv', 
        'ni_catalyzed_2': 'datasets/additive_rxn_screening_plate_2.csv', 
        'ni_catalyzed_3': 'datasets/additive_rxn_screening_plate_3.csv', 
        'ni_catalyzed_4': 'datasets/additive_rxn_screening_plate_4.csv',
    }
    
    file_path = FILEPATH_DICT[dataset]

    if file_path.endswith('.xlsx'):
        lookup = pd.read_excel(file_path, index_col=0 if dataset in ['shields'] else None)
    elif file_path.endswith('.csv'):
        lookup = pd.read_csv(file_path, index_col=0 if dataset in ['shields'] else None)
    else:
        raise ValueError("Unsupported file format. Must be .xlsx or .csv")

    # --- Dataset-specific parsing ---
    if dataset == 'shields':
        # if not args.no_normalize:
        lookup["Temp_C"] = _normalize(lookup["Temp_C"])
        lookup["Concentration"] = _normalize(lookup["Concentration"])

        solvent_data = dict(sorted(set(zip(lookup.Solvent, lookup.Solvent_SMILES))))
        base_data = dict(sorted(set(zip(lookup.Base, lookup.Base_SMILES))))
        ligand_data = dict(sorted(set(zip(lookup.Ligand, lookup.Ligand_SMILES))))

        
        numerical_params = {
            'Temp_C': NumericalDiscreteParameter(name="Temp_C", values=set(lookup.Temp_C)),
            'Concentration': NumericalDiscreteParameter(name="Concentration", values=set(lookup.Concentration)),
        }
        
        discrete_data = {'Solvent': solvent_data,
                        'Base': base_data,
                        'Ligand': ligand_data,
                        'rxn_name': None,
                        }
        

    elif dataset == 'buchwald_hartwig':
        ikeys = lookup.keys()
        for key in ikeys:
            lookup[key+'_name'] = lookup[key]
            
        aryl_halide_data = dict(sorted(set(zip(lookup.aryl_halide_smiles_name, lookup.aryl_halide_smiles))))
        base_data = dict(sorted(set(zip(lookup.base_smiles_name, lookup.base_smiles))))
        ligand_data = dict(sorted(set(zip(lookup.ligand_smiles_name, lookup.ligand_smiles))))
        additive_data = dict(sorted(set(zip(lookup.additive_smiles_name, lookup.additive_smiles))))
        
        numerical_params = {}

        discrete_data = {'aryl_halide_smiles_name': aryl_halide_data,
                        'base_smiles_name': base_data,
                        'ligand_smiles_name': ligand_data,
                        'additive_smiles_name': additive_data,
                        'rxn_name': None,
                        }
        
    elif dataset in ['ni_catalyzed_1', 'ni_catalyzed_2', 'ni_catalyzed_3', 'ni_catalyzed_4']: 
        # NOTE: this task is to maximize UV210 product area absorption. Here we simply treat it as yield. The objective is normalized into [0,100]
        ikeys = lookup.keys()
        for key in ikeys:
            lookup[key+'_name'] = lookup[key]
        lookup['yield'] = _normalize_100(lookup['objective'])
            
        additives_data = dict(sorted(set(zip(lookup.additives_name, lookup.additives))))
        rxn_data = dict(sorted(set(zip(lookup.rxn_name, lookup.rxn))))
        
        discrete_data = {'rxn_name': rxn_data,
                        'additives_name': additives_data}
        
        numerical_params = {}
    
    elif dataset == 'suzuki':
        ikeys = lookup.keys()
        for key in ikeys:
            lookup[key+'_name'] = lookup[key]
            
        electrophile_data = dict(sorted(set(zip(lookup.Electrophile_SMILES_name, lookup.Electrophile_SMILES))))
        nucleophile_data = dict(sorted(set(zip(lookup.Nucleophile_SMILES_name, lookup.Nucleophile_SMILES))))
        ligand_data = dict(sorted(set(zip(lookup.Ligand_SMILES_name, lookup.Ligand_SMILES))))
        base_data = dict(sorted(set(zip(lookup.Base_SMILES_name, lookup.Base_SMILES))))
        solvent_data = dict(sorted(set(zip(lookup.Solvent_SMILES_name, lookup.Solvent_SMILES))))
        
        discrete_data = {'Electrophile_SMILES_name': electrophile_data,
                        'Nucleophile_SMILES_name': nucleophile_data,
                        'Ligand_SMILES_name': ligand_data,
                        'Base_SMILES_name': base_data,
                        'Solvent_SMILES_name': solvent_data,
                        'rxn_name': None,
                        }
        
        numerical_params = {}
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    # lookup['yield'] = _normalize(lookup['yield']) # This normalization doesn't matter because: 
    # (1) we use only one single target;
    # (2) target standardization is automatically handled in BoTorch: see GaussianProcessSurrogate class of BayBE.
    F_BEST = lookup['yield'].max()
    objective = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))


    # --- Return unified structure ---
    return {
        'lookup': lookup,
        'F_BEST': F_BEST,
        'objective': objective,
        'numerical_params': numerical_params,
        'discrete_data': discrete_data,
    }

def generate_FP(data_dict, fp_type, PCA=False, decorr_threshold=0.7):
    """
    data_dict = {
            'solvent': solvent_data | None,
            'base': base_data | None,
            'ligand': ligand_data | None,
            'additive': ...,
            'aryl_halide': ...,
        }
    """
    if fp_type in ['chemeleon', 'CheMeleon']:
        fingerprinter = PretrainedWrapper(CheMeleonFingerprint)
    elif fp_type in ['chemberta_small', 'ChemBERTa_s']:
        fingerprinter = PretrainedWrapper(ChemBERTa_Fingerprint, variant='zinc-base-v1')
    elif fp_type in ['chemberta_large', 'ChemBERTa_l']:
        fingerprinter = PretrainedWrapper(ChemBERTa_Fingerprint, variant='deepchem-100M-MLM')
    elif fp_type in ['t5-base', 'T5']:
        fingerprinter = PretrainedWrapper(LLM_Fingerprint, model_name='t5-base', pooling_method='average', normalize_embeddings=False)
    elif fp_type in ['t5-base-chem', 'T5Chem']:
        fingerprinter = PretrainedWrapper(LLM_Fingerprint, model_name='GT4SD/multitask-text-and-chemistry-t5-base-augm', pooling_method='average', normalize_embeddings=False)
    elif fp_type in ['UAE-Large-V1', 'UAE-Large']:
        fingerprinter = PretrainedWrapper(LLM_Fingerprint, model_name='WhereIsAI/UAE-Large-V1', pooling_method='average', normalize_embeddings=False)
    elif fp_type in ['drfp', 'DRFP']:
        fingerprinter = PretrainedWrapper(Reaction_Fingerprint, variant='drfp')
    else:
        fingerprinter = None
    
    def make_param(name, data):
        if data is None:
            return None
        
        NORMALIZE = None # 'global' | 'local' 
        # NOTE This is not used because BayBE calls BoTorch to handle the input (local) normalization and outcome standardization automatically.
        
        if fp_type in ['mordred', 'Mordred']:
            if PCA:
                return CustomDiscreteParameter(
                    name=name,
                    data=custom_PCA_from_substance(data=data, encoding='MORDRED', norm=NORMALIZE),
                    decorrelate=False,
                )
            else:
                return SubstanceParameter(name=name, data=data, encoding='MORDRED', decorrelate=decorr_threshold) # global normalization for MORDRED not implemented explicitly. This could be done by modifying BayBE source code.
            
        elif fp_type in ['one_hot', 'OHE', 'One-Hot']:
            return CategoricalParameter(name=name, values=data.keys(), encoding="OHE")
        
        else:
            if PCA:
                return CustomDiscreteParameter(
                    name=name,
                    data=custom_PCA_fingerprinter(data, fingerprinter, norm=NORMALIZE),
                    decorrelate=False,
                )
            else:
                return CustomDiscreteParameter(
                    name=name,
                    data=custom_fingerprinter(data, fingerprinter, norm=NORMALIZE),
                    decorrelate=decorr_threshold,
                )
            
    # Automatically handle whatever data keys are available
    if fp_type in ['drfp', 'DRFP']:
        param_dict = {'rxn_name': make_param('rxn_name', data_dict['rxn_name'])}
    else:
        param_dict = {name: make_param(name, data) for name, data in data_dict.items() if data is not None and name!= 'rxn_name'}
    return param_dict
    
def create_recommender(kernel_prior, switch_after, acq_func, searchspace=None, init_method='random', feat_dim=None, kernel_name='Matern'):
    '''
    Give searchspace when LHS is wanted.
    '''
    custom_surrogate = GaussianProcessSurrogate(
        kernel_or_factory= MaternKernelFactory(
            prior_set=kernel_prior,
            n_dim=feat_dim,
            kernel_name_user = kernel_name,
        )
    )
    
    # TODO LHSRecommender's randomness: not perfectly fixed.
    LHS_DICT = {'LHS': LHS1Recommender, 'LHS1': LHS1Recommender, 'LHS2': LHS2Recommender, 'PCALHS': PCALHS2Recommender}
    rec_LHS = None
    if (searchspace is not None) and (init_method in LHS_DICT.keys()):
        # set_random_seed(1337)
        rec_LHS = ChunkingRecommender(
        wrapped_recommender_class=LHS_DICT[init_method],
        wrapped_recommender_kwargs={"optimization": "random-cd"}, # "lloyd", None
        total_init_samples=switch_after,
        searchspace=searchspace)
    
    if switch_after > 0:
        recommender=TwoPhaseMetaRecommender(
                initial_recommender=RandomRecommender() if rec_LHS is None else rec_LHS,
                recommender=BotorchRecommender(
                    surrogate_model=custom_surrogate,
                    acquisition_function=acq_func,
                ),
                switch_after=switch_after
            )
    else:
        recommender=BotorchRecommender(
            surrogate_model=custom_surrogate,
            acquisition_function=acq_func,
            )
    
    return recommender

def create_search_space(search_params_dict, numeric_params_dict):
    search_params_list = []
    
    for name, parameter in search_params_dict.items():
        if parameter is not None:
            search_params_list.append(parameter)
    
    for name, parameter in numeric_params_dict.items():
        search_params_list.append(parameter)
    
    searchspace = SearchSpace.from_product(parameters=search_params_list)
    
    return searchspace

def create_campaign(searchspace, objective, recommender):
    
    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=recommender,
    )
    
    return campaign
