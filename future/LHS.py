"""
Latin Hypercube Sampling (LHS) Recommenders for BayBE.

This module provides three LHS-based recommenders with different strengths:
- LHS1Recommender: strength=1 (simple LHS)
- LHS2RecommenderNextPrime: strength=2 (using next larger prime, truncated)
- LHS2RecommenderPrevPrime: strength=2 (using next smaller prime, filled with random)


## USAGE

# Simple LHS (strength=1)
rec1 = LHS1Recommender(optimization="random-cd")

# Strength=2 with next prime (truncated)
rec2 = LHS2RecommenderNextPrime(optimization="lloyd")

# Strength=2 with previous prime (filled with random)
rec3 = LHS2RecommenderPrevPrime()

# Use in BayBE campaign
recommendations = rec1.recommend(batch_size=10, searchspace=searchspace)

"""

from typing import ClassVar, Literal
import pandas as pd
import numpy as np
from scipy.stats import qmc
import warnings
from typing import Type, Any
from sklearn.decomposition import PCA

from attrs import define, field

from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType
from typing_extensions import override

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# Helper Functions for Prime Number Operations
# ============================================================================

def _is_prime(n: int) -> bool:
    """Check if a number is prime.

    Args:
        n: Integer to check

    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _next_prime(n: int) -> int:
    """Find the next prime number >= n.

    Args:
        n: Starting point

    Returns:
        First prime number >= n
    """
    candidate = n
    while not _is_prime(candidate):
        candidate += 1
    return candidate


def _prev_prime(n: int) -> int:
    """Find the previous prime number <= n.

    Args:
        n: Starting point

    Returns:
        Largest prime number <= n, or 2 if n < 2
    """
    if n < 2:
        return 2
    candidate = n
    while not _is_prime(candidate):
        candidate -= 1
    return candidate


# ============================================================================
# Base Class for LHS Recommenders
# ============================================================================

@define
class LHSRecommenderBase(NonPredictiveRecommender):
    """Base class for LHS Recommenders.

    Provides common initialization and prime-checking logic for all LHS recommenders.
    """

    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID

    # Attrs fields for initialization
    optimization: Literal["random-cd", "lloyd", None] | None = field(default="random-cd")
    enable_auto_fill: bool = field(default=True, kw_only=True)
    auto_fill_warning: bool = field(default=True, kw_only=True)

    def _scale_continuous_samples(
        self,
        searchspace: SearchSpace,
        samples: np.ndarray,
    ) -> pd.DataFrame:
        """Scale LHS samples from [0,1) to parameter bounds.

        Args:
            searchspace: The search space with parameter bounds
            samples: LHS samples in [0,1) with shape (batch_size, n_continuous)

        Returns:
            DataFrame with scaled continuous samples
        """
        params = searchspace.continuous.parameters
        scaled_data = {}

        for i, param in enumerate(params):
            lower = param.bounds.lower
            upper = param.bounds.upper
            scaled_data[param.name] = lower + samples[:, i] * (upper - lower)

        return pd.DataFrame(scaled_data)

    def _map_discrete_samples(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        disc_samples: np.ndarray,
        batch_size: int,
    ) -> pd.DataFrame:
        """Map LHS samples to discrete candidate rows from filtered candidates.

        For each discrete parameter, uses the corresponding LHS dimension to select
        from that parameter's unique values. Then merges with filtered candidates
        to return only those combinations that are valid (not already measured, etc).

        Args:
            searchspace: The search space with discrete parameters
            candidates_exp: Filtered candidate experiments (excludes measured, etc)
            disc_samples: LHS samples in [0,1) with shape (batch_size, n_discrete)
            batch_size: Number of samples

        Returns:
            DataFrame with selected discrete candidate rows, preserving their indices
        """
        discrete_params = searchspace.discrete.parameters
        sample_dict = {}

        # For each discrete parameter, map all LHS samples to parameter values
        for param_idx, param in enumerate(discrete_params):
            param_values = param.values
            n_values = len(param_values)

            # Vectorized mapping: [0,1) to indices
            indices = np.floor(disc_samples[:, param_idx] * n_values).astype(int)
            indices = np.minimum(indices, n_values - 1)

            # Select values from parameter values using indices
            sample_dict[param.name] = [param_values[i] for i in indices]

        # Create DataFrame with selected parameter values
        selected_df = pd.DataFrame(sample_dict)

        # Merge with filtered candidates, preserving their original indices
        # Reset index to make it a column for merging
        candidates_reset = candidates_exp.reset_index()

        # Merge on parameter columns - only returns rows where LHS selections match filtered candidates
        merged = pd.merge(selected_df, candidates_reset, on=list(sample_dict.keys()))

        # Restore the original index from candidates_exp
        if 'index' in merged.columns:
            merged = merged.set_index('index')

        return merged

    def _auto_fill(
        self,
        returned_df: pd.DataFrame,
        batch_size: int,
        candidates_exp: pd.DataFrame,
        searchspace: SearchSpace,
    ) -> pd.DataFrame:
        """Auto-fill missing samples from remaining candidates to reach batch_size.

        If the LHS-selected candidates are fewer than batch_size (e.g., because some
        LHS selections don't match filtered candidates), this method fills the shortage
        with random samples from the remaining candidates using RandomRecommender.

        Args:
            returned_df: DataFrame returned from LHS sampling
            batch_size: Requested number of samples
            candidates_exp: Filtered candidate experiments to fill from
            searchspace: The search space object

        Returns:
            DataFrame with exactly batch_size rows (or fewer if not enough candidates exist)
        """
        if not self.enable_auto_fill or len(returned_df) >= batch_size:
            return returned_df

        shortage = batch_size - len(returned_df)

        if self.auto_fill_warning:
            warnings.warn(
                f"{self.__class__.__name__} returned {len(returned_df)}/{batch_size} samples. "
                f"Auto-filling {shortage} additional samples from remaining candidates.",
                UserWarning
            )

        # Filter to candidates NOT already selected
        already_selected_indices = returned_df.index
        remaining_candidates = candidates_exp.drop(already_selected_indices, errors='ignore')

        # Use RandomRecommender to fill the shortage
        from baybe.recommenders import RandomRecommender
        fill_samples = RandomRecommender()._recommend_hybrid(
            searchspace=searchspace,
            candidates_exp=remaining_candidates,
            batch_size=shortage,
        )

        # Concatenate with returned samples, preserving indices
        result = pd.concat([returned_df, fill_samples], ignore_index=False)

        return result

    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        """Generate LHS recommendations for hybrid search spaces.

        Must be implemented by subclasses.

        Args:
            searchspace: The search space to sample from
            candidates_exp: Candidate experiments
            batch_size: Number of recommendations

        Returns:
            DataFrame with recommended experiments
        """
        raise NotImplementedError("Subclasses must implement _recommend_hybrid")

    @override
    def __str__(self) -> str:
        """String representation of the recommender."""
        opt_str = f"optimization={self.optimization}"
        return f"{self.__class__.__name__}(strength={self.strength}, {opt_str})"


# ============================================================================
# LHS1Recommender: strength=1
# ============================================================================

@define
class LHS1Recommender(LHSRecommenderBase):
    """LHS Recommender with strength=1.

    Strength=1 is the standard Latin Hypercube Sampling with no special
    orthogonality properties. Works with any number of dimensions.
    """

    @override
    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        """Generate LHS1 recommendations for hybrid search spaces."""
        # Determine dimensionality: continuous dims + discrete dims
        n_continuous = len(searchspace.continuous.parameters)
        n_discrete = len(searchspace.discrete.parameters) if searchspace.discrete else 0
        n_dims = n_continuous + n_discrete

        # Generate LHS samples in [0, 1)
        sampler = qmc.LatinHypercube(d=n_dims, optimization=self.optimization)
        lhs_samples = sampler.random(n=batch_size)

        # Handle pure continuous case
        if searchspace.type == SearchSpaceType.CONTINUOUS:
            return self._scale_continuous_samples(searchspace, lhs_samples)

        # Handle pure discrete case
        if searchspace.type == SearchSpaceType.DISCRETE:
            return self._map_discrete_samples(searchspace, candidates_exp, lhs_samples, batch_size)

        # Handle hybrid case
        # Split into continuous and discrete parts
        cont_samples = lhs_samples[:, :n_continuous]
        disc_samples = lhs_samples[:, n_continuous:n_continuous + n_discrete]

        # Process parts
        cont_df = self._scale_continuous_samples(searchspace, cont_samples)
        disc_df = self._map_discrete_samples(searchspace, candidates_exp, disc_samples, batch_size)

        # Align and concatenate
        cont_df.index = disc_df.index
        return pd.concat([disc_df, cont_df], axis=1)

# ============================================================================
# LHS2Recommender: strength=2 (fill with random)
# ============================================================================

@define
class LHS2Recommender(LHSRecommenderBase):
    """LHS Recommender with strength=2 using scipy's orthogonal array sampling.
    
    Uses scipy's strength=2 parameter which requires n > (d-1)^2 where n is batch_size
    and d is dimensions. Generates LHS samples and auto-fills to reach batch_size.
    """

    @override
    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        """Generate LHS2 recommendations using strength=2 orthogonal arrays."""
        n_continuous = len(searchspace.continuous.parameters)
        n_discrete = len(searchspace.discrete.parameters) if searchspace.discrete else 0
        n_dims = n_continuous + n_discrete
        prime = int(np.sqrt(batch_size))
        if _is_prime(prime):
            n_samples = prime**2
        else:
            n_samples = _prev_prime(prime)**2

        try:
            sampler = qmc.LatinHypercube(d=n_dims, strength=2, optimization=self.optimization)
            lhs_samples = sampler.random(n=n_samples)
        except:
            min_samples_needed = (prime + 1) ** 2
            warnings.warn(
                f"{self.__class__.__name__}: batch_size={batch_size} results in n_samples={n_samples} "
                f"which is below minimum {min_samples_needed} for strength=2. "
                f"Falling back to strength=1 LHS with auto_fill.",
                UserWarning
            )
            # Fall back to strength=1 with auto_fill
            sampler = qmc.LatinHypercube(d=n_dims, strength=1, optimization=self.optimization)
            lhs_samples = sampler.random(n=batch_size)

        # Handle pure continuous case
        if searchspace.type == SearchSpaceType.CONTINUOUS:
            cont_df = self._scale_continuous_samples(searchspace, lhs_samples)
            return self._auto_fill(cont_df, batch_size, candidates_exp, searchspace)

        # Handle pure discrete case
        if searchspace.type == SearchSpaceType.DISCRETE:
            disc_df = self._map_discrete_samples(searchspace, candidates_exp, lhs_samples, batch_size)
            return self._auto_fill(disc_df, batch_size, candidates_exp, searchspace)

        # Handle hybrid case
        # Split into continuous and discrete parts
        cont_samples = lhs_samples[:, :n_continuous]
        disc_samples = lhs_samples[:, n_continuous:n_continuous + n_discrete]

        # Process parts
        cont_df = self._scale_continuous_samples(searchspace, cont_samples)
        disc_df = self._map_discrete_samples(searchspace, candidates_exp, disc_samples, batch_size)

        # Align and concatenate BEFORE auto-fill
        cont_df.index = disc_df.index
        result = pd.concat([disc_df, cont_df], axis=1)

        # Auto-fill if needed on final concatenated result
        return self._auto_fill(result, batch_size, candidates_exp, searchspace)


# ============================================================================
# PCALHS2Recommender: strength=2 (fill with random)
# ============================================================================

@define
class PCALHS2Recommender(LHSRecommenderBase):
    """LHS Recommender with strength=2 + PCA on discrete dimensions.
    
    Applies PCA dimensionality reduction to discrete parameters when needed,
    then uses strength=2 LHS in the reduced space. Continuous parameters 
    are handled normally without PCA.
    """

    @override
    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        """Generate LHS2 recommendations with PCA on discrete dimensions."""

        n_continuous = len(searchspace.continuous.parameters)
        if searchspace.discrete:
            searchspace_exp, searchspace_comp = searchspace.discrete.get_candidates()
            n_discrete = searchspace_comp.shape[1]
            
            # For each row in candidates_exp, find the matching row in searchspace_exp
            # Then get the corresponding row from searchspace_comp
            candidates_comp_rows = []
            candidates_comp_index = []
            
            for cand_idx, cand_row in candidates_exp.iterrows():
                # Find matching row in searchspace_exp (same discrete parameter values)
                match = searchspace_exp.eq(cand_row).all(axis=1)
                match_idx = match.idxmax()  # Get the index of the matching row
                
                if match.any():
                    # Get corresponding row from searchspace_comp
                    candidates_comp_rows.append(searchspace_comp.loc[match_idx].values)
                    candidates_comp_index.append(cand_idx)
            
            # Create candidates_comp with candidates_exp indices
            candidates_comp = pd.DataFrame(
                candidates_comp_rows,
                index=candidates_comp_index,
                columns=searchspace_comp.columns
            )
        else:
            n_discrete = 0
            candidates_comp = None
            
        prime = int(np.sqrt(batch_size))
        if _is_prime(prime):
            n_samples = prime**2
        else:
            n_samples = _prev_prime(prime)**2
        
        # Determine if PCA is needed for discrete dimensions
        n_discrete_max = prime + 1 - n_continuous
        pca = None
        do_pca = False
        
        if n_discrete > 0 and n_discrete > n_discrete_max:
            # Apply PCA to reduce discrete dimensionality
            do_pca = True
            pca = PCA(n_components=n_discrete_max)
            candidates = pca.fit_transform(candidates_comp)
            n_dims = prime+1
        else:
            candidates = candidates_exp
            n_dims = n_discrete + n_continuous
      
        sampler = qmc.LatinHypercube(d=n_dims, strength=2, optimization=self.optimization)
        lhs_samples = sampler.random(n=n_samples)

        # Handle pure continuous case
        if searchspace.type == SearchSpaceType.CONTINUOUS:
            cont_df = self._scale_continuous_samples(searchspace, lhs_samples)
            return self._auto_fill(cont_df, batch_size, candidates_exp, searchspace)

        # Handle pure discrete case
        if searchspace.type == SearchSpaceType.DISCRETE:
            scaler = MinMaxScaler()
            scaler.fit(candidates)

            lhs_samples_comp = scaler.inverse_transform(lhs_samples)
            if do_pca:
                lhs_samples_comp = pca.inverse_transform(lhs_samples_comp)
            # Convert to DataFrame with the same column names as candidates_comp
            lhs_samples_comp = pd.DataFrame(
                lhs_samples_comp, 
                columns=candidates_comp.columns)

            nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn.fit(candidates_comp)  # Shape: (1728, 15)
            distances, nn_indices = nn.kneighbors(lhs_samples_comp)  # lhs_samples shape: (9, 15)
            # Map back to the DataFrame rows
            #closest_candidates_comp = candidates_comp.iloc[nn_indices.flatten()]
            #closest_candidates_exp = candidates_exp.iloc[nn_indices.flatten()]
            #actual_indices = closest_candidates_comp.index.values        
            disc_df = candidates_exp.iloc[nn_indices.flatten()]
            return self._auto_fill(disc_df, batch_size, candidates_exp, searchspace)

        # Handle hybrid case
        # Split into continuous and discrete parts
        cont_samples = lhs_samples[:, :n_continuous]
        disc_lhs_samples = lhs_samples[:, n_continuous:n_continuous + n_discrete_max]

        scaler = MinMaxScaler()
        scaler.fit(candidates)

        lhs_samples_comp = scaler.inverse_transform(disc_lhs_samples)
        if do_pca:
            lhs_samples_comp = pca.inverse_transform(lhs_samples_comp)
        # Convert to DataFrame with the same column names as candidates_comp
        lhs_samples_comp = pd.DataFrame(
            lhs_samples_comp, 
            columns=candidates_comp.columns)

        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(candidates_comp)  # Shape: (1728, 15)
        distances, nn_indices = nn.kneighbors(lhs_samples_comp)  # lhs_samples shape: (9, 15)
        # Map back to the DataFrame rows
        #closest_candidates_comp = candidates_comp.iloc[nn_indices.flatten()]
        #closest_candidates_exp = candidates_exp.iloc[nn_indices.flatten()]
        #actual_indices = closest_candidates_comp.index.values        
        disc_df = candidates_exp.iloc[nn_indices.flatten()]

        # Process parts
        cont_df = self._scale_continuous_samples(searchspace, cont_samples)

        # Align and concatenate BEFORE auto-fill
        cont_df.index = disc_df.index
        result = pd.concat([disc_df, cont_df], axis=1)

        # Auto-fill if needed on final concatenated result
        return self._auto_fill(result, batch_size, candidates_exp, searchspace)

@define
class ChunkingRecommender(NonPredictiveRecommender):
    """Chunks large initialization batches into smaller recommendation batches."""
    
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    
    wrapped_recommender_class: Type[NonPredictiveRecommender] = field()
    wrapped_recommender_kwargs: dict[str, Any] = field()
    total_init_samples: int = field()
    searchspace: SearchSpace = field()
    candidates_exp: pd.DataFrame | None = field(default=None)
    _buffer: pd.DataFrame = field(init=False, default=None)
    _buffer_idx: int = field(init=False, default=0)    
    
    def _resolve_candidates_exp(self) -> pd.DataFrame:
        """Get candidates_exp from field or generate from searchspace if not provided."""
        if self.candidates_exp is not None:
            return self.candidates_exp
        disc_candidates, _ = self.searchspace.discrete.get_candidates()
        return disc_candidates
    
    def _init_buffer(self):
        """Initialize buffer on first recommendation."""
        wrapped_recommender = self.wrapped_recommender_class(**self.wrapped_recommender_kwargs)
        resolved_candidates = self._resolve_candidates_exp()
        
        self._buffer = wrapped_recommender._recommend_hybrid(
            searchspace=self.searchspace,
            candidates_exp=resolved_candidates,
            batch_size=self.total_init_samples,
        )
        self._buffer_idx = 0
    
    @override
    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        # Initialize buffer on first call
        if self._buffer is None:
            self._init_buffer()
        
        # Return next chunk from buffer
        start = self._buffer_idx
        end = min(start + batch_size, len(self._buffer))
        chunk = self._buffer.iloc[start:end]
        self._buffer_idx = end
        return chunk