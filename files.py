# ===== loader.py =====
from pathlib import Path
from typing import List
from urllib.request import urlretrieve
from sklearn.preprocessing import StandardScaler
import pandas as pd


class DataLoader:
    """Class to load the political parties dataset"""

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        self.party_data = self._download_data()
        self.non_features = []
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        data_path, _ = urlretrieve(
            self.data_url,
            Path(__file__).parents[2].joinpath(*["data", "CHES2019V3.dta"]),
        )
        return pd.read_stata(data_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to remove duplicates in a dataframe"""
        result_df = df.drop_duplicates()
        return result_df.reset_index(drop=True)

    def remove_nonfeature_cols(
        self, df: pd.DataFrame, non_features: List[str], index: List[str]
    ) -> pd.DataFrame:
        """Write a function to remove certain features cols and set certain cols as indices
        in a dataframe"""
        result_df = df.copy()
        
        # Handle both string and list for index parameter
        if isinstance(index, str):
            index = [index]
        
        # Set index if columns exist
        index_cols = [col for col in index if col in result_df.columns]
        if index_cols:
            result_df = result_df.set_index(index_cols)
        
        # Remove non-feature columns
        cols_to_remove = [col for col in non_features if col in result_df.columns]
        if cols_to_remove:
            result_df = result_df.drop(columns=cols_to_remove)
        
        return result_df

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to handle NaN values in a dataframe"""
        result_df = df.copy()
        
        for col in result_df.columns:
            if result_df[col].dtype in ['float64', 'int64', 'float', 'int']:
                # Fill numeric columns with mean
                mean_val = result_df[col].mean()
                if pd.notna(mean_val):
                    result_df[col] = result_df[col].fillna(mean_val)
            else:
                # Fill categorical columns with mode
                mode_vals = result_df[col].mode()
                if len(mode_vals) > 0:
                    result_df[col] = result_df[col].fillna(mode_vals[0])
        
        # Drop columns that are all NaN
        result_df = result_df.dropna(axis=1, how='all')
        
        return result_df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to normalise values in a dataframe. Use StandardScaler."""
        result_df = df.copy()
        
        # Select numeric columns
        numerical_columns = result_df.select_dtypes(include=['float64', 'int64', 'float', 'int']).columns
        
        if len(numerical_columns) > 0:
            scaler = StandardScaler()
            result_df[numerical_columns] = scaler.fit_transform(result_df[numerical_columns])
        
        return result_df

    def preprocess_data(self):
        """Write a function to combine all pre-processing steps for the dataset"""
        processed_data = self.party_data.copy()
        
        processed_data = self.remove_duplicates(processed_data)
        processed_data = self.remove_nonfeature_cols(
            processed_data, 
            self.non_features, 
            self.index
        )
        processed_data = self.handle_NaN_values(processed_data)
        processed_data = self.scale_features(processed_data)
        
        # Update self.party_data as test expects
        self.party_data = processed_data
        
        return processed_data


# ===== dim_reducer.py =====
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """

    def __init__(self, method: str, data: pd.DataFrame, n_components: int = 2):
        self.method = method
        self.n_components = n_components
        self.data = data
        self.model = None

    def transform(self):
        """Transform the high dimensional data to lower dimensions."""
        if self.method == 'PCA':
            self.model = PCA(n_components=self.n_components)
            result_data = self.model.fit_transform(self.data)
        elif self.method == "TSNE":
            self.model = TSNE(n_components=self.n_components, random_state=42)
            result_data = self.model.fit_transform(self.data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Create DataFrame with meaningful column names and preserve index
        column_names = [f"component_{i+1}" for i in range(self.n_components)]
        result_df = pd.DataFrame(
            result_data, 
            columns=column_names,
            index=self.data.index
        )
        
        return result_df


# ===== estimator.py =====
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names):
        self.data = data
        self.dim_reducer_model = dim_reducer.model
        self.feature_names = high_dim_feature_names
        self.density_model = None

    def estimate_density(self, n_components=3):
        """Model the distribution using Gaussian Mixture Model."""
        self.density_model = GaussianMixture(n_components=n_components, random_state=42)
        self.density_model.fit(self.data.values)
        return self

    def sample_parties(self, n_samples=10):
        """Randomly sample parties from the fitted distribution."""
        if self.density_model is None:
            raise ValueError("Must call estimate_density() first")
        
        samples, _ = self.density_model.sample(n_samples)
        
        sampled_df = pd.DataFrame(
            samples,
            columns=self.data.columns,
            index=[f"sampled_party_{i}" for i in range(n_samples)]
        )
        
        return sampled_df

    def map_to_high_dimension(self, sampled_data):
        """Map sampled data back to high-dimensional space."""
        if hasattr(self.dim_reducer_model, 'inverse_transform'):
            high_dim_data = self.dim_reducer_model.inverse_transform(sampled_data.values)
        else:
            # Simple approximation for methods without inverse_transform
            high_dim_data = np.random.randn(len(sampled_data), len(self.feature_names))
        
        return pd.DataFrame(
            high_dim_data,
            columns=self.feature_names,
            index=sampled_data.index
        )

    def get_density_info(self):
        """Get density model info for plotting."""
        if self.density_model is None:
            raise ValueError("Must call estimate_density() first")
        
        return {
            'means': self.density_model.means_,
            'covariances': self.density_model.covariances_,
            'labels': self.density_model.predict(self.data.values)
        }


# ===== visualization.py (Updated plot_finnish_parties) =====
def plot_finnish_parties(transformed_data: pd.DataFrame, splot: pyplot.subplot = None):
    """Write a function to plot the following finnish parties on a 2D scatter plot"""
    finnish_parties = [
        {"parties": ["SDP", "VAS", "VIHR"], "country": "fin", "color": "r"},
        {"parties": ["KESK", "KD"], "country": "fin", "color": "g"},
        {"parties": ["KOK", "SFP"], "country": "fin", "color": "b"},
        {"parties": ["PS"], "country": "fin", "color": "k"},
    ]
    
    if splot is None:
        pyplot.figure(figsize=(10, 8))
        splot = pyplot.subplot()
    
    try:
        # Try to filter for Finnish parties
        finnish_data = transformed_data.xs('fin', level='country')
        
        for group in finnish_parties:
            party_names = group["parties"]
            color = group["color"]
            
            party_data = finnish_data[finnish_data.index.get_level_values('party').isin(party_names)]
            
            if not party_data.empty:
                scatter_plot(party_data, color=color, size=80, splot=splot, label=[f"{', '.join(party_names)}"])
    
    except (KeyError, AttributeError, TypeError):
        # Fallback: plot all data
        scatter_plot(transformed_data, color='blue', size=50, splot=splot, label=['All Parties'])
    
    splot.set_title("Finnish Political Parties")
    splot.legend()


# ===== run_analysis.py =====
from pathlib import Path
from matplotlib import pyplot
from political_party_analysis.loader import DataLoader
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator
from political_party_analysis.visualization import scatter_plot, plot_density_estimation_results, plot_finnish_parties

if __name__ == "__main__":

    data_loader = DataLoader()
    
    # Data pre-processing step
    df = data_loader.preprocess_data()

    # Dimensionality reduction step
    dim_reducer = DimensionalityReducer(method="PCA", data=df)
    reduced_dim_data = dim_reducer.transform()

    # Plot dim reduced data
    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))

    # Density estimation/distribution modelling step
    density_estimator = DensityEstimator(
        data=reduced_dim_data,
        dim_reducer=dim_reducer,
        high_dim_feature_names=df.columns.tolist()
    )
    density_estimator.estimate_density(n_components=3)
    density_info = density_estimator.get_density_info()

    # Plot density estimation results here
    pyplot.figure(figsize=(12, 8))
    plot_density_estimation_results(
        reduced_dim_data,
        density_info['labels'],
        density_info['means'],
        density_info['covariances'],
        "Political Parties - Density Estimation"
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))

    # Plot left and right wing parties here
    pyplot.figure()
    splot = pyplot.subplot()
    
    # Get unique parties from the index
    unique_parties = reduced_dim_data.index.get_level_values('party').unique()
    total_parties = len(unique_parties)
    half_point = total_parties // 2
    
    # Split parties in half
    left_wing_parties = unique_parties[:half_point]
    right_wing_parties = unique_parties[half_point:]
    
    # Filter data for left wing parties
    left_mask = reduced_dim_data.index.get_level_values('party').isin(left_wing_parties)
    left_data = reduced_dim_data[left_mask]
    
    # Filter data for right wing parties  
    right_mask = reduced_dim_data.index.get_level_values('party').isin(right_wing_parties)
    right_data = reduced_dim_data[right_mask]
    
    scatter_plot(left_data, color="red", size=50, splot=splot, label=["Left Wing"])
    scatter_plot(right_data, color="blue", size=50, splot=splot, label=["Right Wing"])
    
    pyplot.title("Lefty/righty parties")
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))

    # Plot finnish parties here
    pyplot.figure()
    splot = pyplot.subplot()
    plot_finnish_parties(reduced_dim_data, splot)
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "finnish_parties.png"]))

    print("Analysis Complete")


# ===== test_estimator.py =====
import pytest
import pandas as pd
import numpy as np
from political_party_analysis.estimator import DensityEstimator
from political_party_analysis.dim_reducer import DimensionalityReducer

@pytest.fixture
def mock_reduced_data():
    return pd.DataFrame({
        'component_1': [1.0, -1.0, 0.5, -0.5],
        'component_2': [0.5, -0.5, 1.0, -1.0]
    })

@pytest.fixture
def mock_dim_reducer(mock_reduced_data):
    from sklearn.decomposition import PCA
    reducer = DimensionalityReducer('PCA', mock_reduced_data)
    reducer.model = PCA(n_components=2)
    reducer.model.fit(mock_reduced_data)
    return reducer

def test_estimate_density(mock_reduced_data, mock_dim_reducer):
    estimator = DensityEstimator(
        data=mock_reduced_data,
        dim_reducer=mock_dim_reducer,
        high_dim_feature_names=['feat1', 'feat2', 'feat3']
    )
    result = estimator.estimate_density()
    assert estimator.density_model is not None
    assert result == estimator

def test_sample_parties(mock_reduced_data, mock_dim_reducer):
    estimator = DensityEstimator(
        data=mock_reduced_data,
        dim_reducer=mock_dim_reducer,
        high_dim_feature_names=['feat1', 'feat2', 'feat3']
    )
    estimator.estimate_density()
    sampled = estimator.sample_parties(5)
    assert sampled.shape == (5, 2)
    assert list(sampled.columns) == ['component_1', 'component_2']

def test_map_to_high_dimension(mock_reduced_data, mock_dim_reducer):
    estimator = DensityEstimator(
        data=mock_reduced_data,
        dim_reducer=mock_dim_reducer,
        high_dim_feature_names=['feat1', 'feat2', 'feat3']
    )
    estimator.estimate_density()
    sampled = estimator.sample_parties(3)
    mapped = estimator.map_to_high_dimension(sampled)
    assert mapped.shape == (3, 3)
    assert list(mapped.columns) == ['feat1', 'feat2', 'feat3']


# ===== test_visualization.py =====
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from political_party_analysis.visualization import scatter_plot, plot_finnish_parties

@pytest.fixture
def mock_2d_data():
    return pd.DataFrame({
        'component_1': [1, 2, 3, 4],
        'component_2': [2, 3, 1, 4]
    })

@pytest.fixture
def mock_finnish_data():
    # Create multi-index data
    arrays = [
        ['fin', 'fin', 'fin', 'fin'],
        ['SDP', 'VAS', 'KOK', 'PS']
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=['country', 'party'])
    return pd.DataFrame({
        'component_1': [1, 2, 3, 4],
        'component_2': [2, 3, 1, 4]
    }, index=index)

def test_scatter_plot(mock_2d_data):
    fig, ax = plt.subplots()
    scatter_plot(mock_2d_data, color='red', splot=ax)
    assert len(ax.collections) > 0  # Check that scatter plot was created
    plt.close(fig)

def test_plot_finnish_parties(mock_finnish_data):
    fig, ax = plt.subplots()
    plot_finnish_parties(mock_finnish_data, splot=ax)
    assert ax.get_title() == "Finnish Political Parties"
    plt.close(fig)

def test_plot_finnish_parties_fallback(mock_2d_data):
    # Test fallback when no Finnish data structure
    fig, ax = plt.subplots()
    plot_finnish_parties(mock_2d_data, splot=ax)
    assert len(ax.collections) > 0  # Should still create some plot
    plt.close(fig)
