import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import Rotator


def generate_rotated_pca_df(features_df, target_df):
    
    pca = PCA(n_components=0.90)
    pca_data = pca.fit_transform(features_df)
    num_components = pca.n_components_
    explained_variance = pca.explained_variance_ratio_

    print(f'Number of selected components: {num_components}')
    print(f'Explained variance by component: {explained_variance}')

    original_features = features_df.columns
    pca_components = pd.DataFrame(
        pca.components_, columns=original_features, index=[f'PC{i+1}' for i in range(num_components)])

    pca_loadings = pca_components.values

    # Rotation
    rotator = Rotator(method='oblimin')
    rotated_loadings = rotator.fit_transform(pca_loadings)
    rotated_pca_components = pd.DataFrame(
        rotated_loadings,
        columns=pca_components.columns,
        index=pca_components.index
    )

    rotated_pca_data = np.dot(features_df, rotated_loadings.T)

    rotated_pca_features_df = pd.DataFrame(
        rotated_pca_data,
        columns=[f'Rotated_PC{i+1}' for i in range(rotated_pca_data.shape[1])]
    )

    rotated_final_df = pd.concat([rotated_pca_features_df, target_df.reset_index(drop=True)], axis=1)
    
    return rotated_final_df