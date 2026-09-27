#Freeze_models is file we train models and save results in Artifacts
#We use most funcitons from main.py but -evaluation so it's faster

import os
import joblib
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors

#get funcitons and dataframes we need from from main.py
from main import (
    load_data, clean_games, split_data, build_interaction_matrix,
    build_content_features, make_als, LIGHTFM_AVAILABLE
)

if LIGHTFM_AVAILABLE:
    from lightfm import LightFM
    from lightfm.data import Dataset as LightFMDataset
    from scipy.sparse import csr_matrix as lightfm_csr


#Save models
def freeze_all_models(output_dir='model_artifacts'):
    os.makedirs(output_dir, exist_ok=True)

    #Get data
    print('Loading and preparing data...')
    games_df, games_metadata_df, interactions_df = load_data()
    games_clean = clean_games(games_df)
    train_df, test_df, test_positive = split_data(interactions_df)

    X, user_mapper, game_mapper, user_inv_mapper, game_inv_mapper, app_ids = \
        build_interaction_matrix(interactions_df, train_df)

    content_df, game_features, content_idx_map, content_inv_map = \
        build_content_features(games_clean, games_metadata_df)

    games_lookup = games_df[['app_id', 'title']].set_index('app_id')['title'].to_dict()

    #training individual models

    #collaborative_knn 
    print('Fitting collaborative kNN...')
    knn_collab_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_collab_model.fit(X)

    #content_based
    print('Fitting content-based kNN...')
    feature_matrix = csr_matrix(game_features.astype(np.float32).values)
    knn_content_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_content_model.fit(feature_matrix)

    #als 
    print('Fitting ALS...')
    _, als_model = make_als(X, game_mapper, game_inv_mapper, factors=20, regularization=0.01, alpha=15)

    #opularity based
    print('Building popularity ranking...')
    popularity_ranking = (
        train_df[train_df['is_recommended'] == True]
        .groupby('app_id')
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    artifacts = {
        'game_mapper': game_mapper,
        'game_inv_mapper': game_inv_mapper,
        'content_idx_map': content_idx_map,
        'content_inv_map': content_inv_map,
        'knn_collab_model': knn_collab_model,
        'knn_content_model': knn_content_model,
        'als_model': als_model,
        'popularity_ranking': popularity_ranking,
        'games_lookup': games_lookup,
        'available_models': ['collaborative_knn', 'collaborative_cosine', 'content_based', 'als', 'popularity'],
    }

    # lightfm
    if LIGHTFM_AVAILABLE:
        print('Fitting LightFM...')
        lightfm_dataset = LightFMDataset()
        lightfm_dataset.fit(
            users=train_df['user_id'].unique(),
            items=train_df['app_id'].unique()
        )
        interactions, weights = lightfm_dataset.build_interactions(
            [(row['user_id'], row['app_id'])
             for _, row in train_df[train_df['is_recommended'] == True].iterrows()]
        )

        item_id_map = lightfm_dataset.mapping()[2]
        inv_item_map = {v: k for k, v in item_id_map.items()}

        feature_matrix_rows = []
        for app_id in item_id_map.keys():
            row_idx = content_idx_map.get(app_id)
            if row_idx is not None:
                feature_matrix_rows.append(game_features.iloc[row_idx].values)
            else:
                feature_matrix_rows.append(np.zeros(game_features.shape[1]))
        item_features_matrix = lightfm_csr(np.array(feature_matrix_rows, dtype=np.float32))

        lightfm_model = LightFM(loss='warp', no_components=50, learning_rate=0.05, random_state=42)
        lightfm_model.fit(interactions, item_features=item_features_matrix, epochs=20, num_threads=4)

        _, item_embeddings = lightfm_model.get_item_representations(features=item_features_matrix)


        artifacts['lightfm_item_embeddings'] = item_embeddings
        artifacts['lightfm_item_id_map'] = item_id_map
        artifacts['lightfm_inv_item_map'] = inv_item_map
        artifacts['available_models'].append('lightfm')
        artifacts['available_models'].append('hybrid')

    print(f'Saving artifacts to {output_dir}/model_artifacts.joblib ...')
    joblib.dump(artifacts, os.path.join(output_dir, 'model_artifacts.joblib'))
    print('Done. Available models:', artifacts['available_models'])


if __name__ == '__main__':
    freeze_all_models()
