import os
from datetime import datetime
from collections import Counter

import kagglehub
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, save_npz
from scipy.sparse import csr_matrix as lightfm_csr

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from rapidfuzz import process

from implicit.als import AlternatingLeastSquares

import json


try:
    from lightfm import LightFM
    from lightfm.data import Dataset as LightFMDataset
    LIGHTFM_AVAILABLE = True
except ImportError:
    LIGHTFM_AVAILABLE = False
    print('lightfm not installed -- skipping that model. See install notes if you want it enabled.')


# 1. Data loading
def load_data():
    path = kagglehub.dataset_download('antonkozyriev/game-recommendations-on-steam')
    print('Path to dataset files:', path)

    games_raw = pd.read_csv(os.path.join(path, 'games.csv'))
    users_raw = pd.read_csv(os.path.join(path, 'users.csv'))
    recommendations_raw = pd.read_csv(os.path.join(path, 'recommendations.csv'))
    games_metadata_df = pd.read_json(os.path.join(path, 'games_metadata.json'), lines=True)

    games_df = games_raw.copy()
    interactions_df = recommendations_raw.merge(users_raw, on='user_id', how='left')

    interactions_df['date'] = pd.to_datetime(interactions_df['date'], errors='coerce')
    interactions_df = interactions_df.sort_values('date')

    return games_df, games_metadata_df, interactions_df


# 2. Data cleaning(EDA)
def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, TypeError):
        return None


def clean_games(games_df):
    games_clean = games_df.copy()

    #drop useless columns
    games_clean = games_clean.drop(
        columns=['win', 'mac', 'linux', 'steam_deck', 'discount', 'price_original', 'title'],
        errors='ignore'
    )

    #convert to datetime
    games_clean['date_release'] = games_clean['date_release'].apply(parse_date)

    print('Rating categories:', games_clean['rating'].unique())

    #convert ratings to int
    rating_map = {
        'Overwhelmingly Negative': 0,
        'Very Negative': 1,
        'Negative': 2,
        'Mostly Negative': 3,
        'Mixed': 4,
        'Mostly Positive': 5,
        'Positive': 6,
        'Very Positive': 7,
        'Overwhelmingly Positive': 8
    }
    games_clean['rating'] = games_clean['rating'].map(rating_map)

    before = games_clean.shape[0]
    games_clean = games_clean[games_clean['user_reviews'] >= 10]
    after = games_clean.shape[0]
    print(f"Dropped {before - after} games with <10 reviews ({before} -> {after})")

    return games_clean


# 3. Train/test split
#80/20
def split_data(interactions_df):
    split_idx = int(len(interactions_df) * 0.8)
    train_df = interactions_df.iloc[:split_idx].copy()
    test_df = interactions_df.iloc[split_idx:].copy()
    test_positive = test_df[test_df['is_recommended'] == True]
    return train_df, test_df, test_positive


# 4. Interaction matrix (collaborative filtering input)

def build_interaction_matrix(interactions_df, train_df):
    user_ids = interactions_df['user_id'].unique()
    app_ids = interactions_df['app_id'].unique()

    user_mapper = {uid: i for i, uid in enumerate(user_ids)}
    game_mapper = {aid: i for i, aid in enumerate(app_ids)}

    user_inv_mapper = {i: uid for uid, i in user_mapper.items()}
    game_inv_mapper = {i: aid for aid, i in game_mapper.items()}

    train_df['user_idx'] = train_df['user_id'].map(user_mapper)
    train_df['app_idx'] = train_df['app_id'].map(game_mapper)

    X = csr_matrix(
        (train_df['is_recommended'].astype(int),
         (train_df['app_idx'], train_df['user_idx'])),
        shape=(len(game_mapper), len(user_mapper))
    )

    sparsity = X.count_nonzero() / (X.shape[0] * X.shape[1])
    print(f'Matrix sparsity: {round(sparsity * 100, 2)}%')

    os.makedirs('data', exist_ok=True)
    save_npz('data/user_item_matrix.npz', X)

    return X, user_mapper, game_mapper, user_inv_mapper, game_inv_mapper, app_ids


# 5. Content-based features

def build_content_features(games_clean, games_metadata_df):
    content_df = games_clean.merge(
        games_metadata_df[['app_id', 'tags']], on='app_id', how='left'
    )
    content_df['tags'] = content_df['tags'].apply(lambda x: x if isinstance(x, list) else [])

    tag_counts = Counter(t for tags in content_df['tags'] for t in tags)
    print(f'There are {len(tag_counts)} unique tags.')
    print('Top 10 tags:', tag_counts.most_common(10))

    top_tags = [t for t, _ in tag_counts.most_common(100)]
    for t in top_tags:
        content_df[t] = content_df['tags'].apply(lambda x: int(t in x))

    content_df['year'] = content_df['date_release'].apply(lambda d: d.year if d is not None else None)

    def round_down(year):
        return year - (year % 10)

    content_df['decade'] = content_df['year'].apply(lambda y: round_down(y) if y is not None else None)
    decade_dummies = pd.get_dummies(content_df['decade'], prefix='decade')

    content_df['price_bucket'] = pd.cut(
        content_df['price_final'],
        bins=[-0.01, 0, 5, 10, 20, 30, 60, np.inf],
        labels=['free', '0-5', '5-10', '10-20', '20-30', '30-60', '60+']
    )
    price_dummies = pd.get_dummies(content_df['price_bucket'], prefix='price')

    game_features = pd.concat(
        [content_df[top_tags], decade_dummies, price_dummies],
        axis=1
    ).fillna(0)

    print('game_features shape:', game_features.shape)

    content_idx_map = dict(zip(content_df['app_id'], content_df.index))
    content_inv_map = dict(zip(content_df.index, content_df['app_id']))

    return content_df, game_features, content_idx_map, content_inv_map


def game_finder(title, games_lookup):
    all_titles = games_lookup['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]


# KNN Neighbors

def make_collaborative_knn(X, game_mapper, game_inv_mapper):
    def _recommend(app_id, k=10, metric='cosine'):
        game_ind = game_mapper[app_id]
        game_vec = X[game_ind]
        if isinstance(game_vec, np.ndarray):
            game_vec = game_vec.reshape(1, -1)

        kNN = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric=metric)
        kNN.fit(X)
        distances, indices = kNN.kneighbors(game_vec, return_distance=True)

        return [game_inv_mapper[indices[0, i]] for i in range(1, k + 1)]
    return _recommend


#Colladb. cosin (same as KNN just with cosin instead)
def make_collaborative_cosine(cosine_sim, game_mapper, game_inv_mapper):
    def _recommend(app_id, k=10):
        game_ind = game_mapper[app_id]
        sim_scores = list(enumerate(cosine_sim[game_ind]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k + 1]
        return [game_inv_mapper[i] for i, _ in sim_scores]
    return _recommend


def make_content_based(game_features, content_idx_map, content_inv_map):
    

    feature_matrix = csr_matrix(game_features.astype(np.float32).values)

    knn_content = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_content.fit(feature_matrix)

    def _recommend(app_id, k=10):
        idx = content_idx_map[app_id]
        game_vec = feature_matrix[idx]
        distances, indices = knn_content.kneighbors(game_vec, n_neighbors=k + 1)
        return [content_inv_map[i] for i in indices[0] if i != idx][:k]

    return _recommend




def make_als(X, game_mapper, game_inv_mapper, train_df, test_positive):
   
    param_grid = [
        {'factors': 20, 'regularization': 0.01, 'alpha': 1},
        {'factors': 20, 'regularization': 0.01, 'alpha': 15},
        {'factors': 50, 'regularization': 0.01, 'alpha': 15},
        {'factors': 50, 'regularization': 0.1, 'alpha': 15},
        {'factors': 50, 'regularization': 0.1, 'alpha': 40},
        {'factors': 100, 'regularization': 0.1, 'alpha': 40},
    ]

    best_score = -1
    best_params = None
    best_model = None
    all_runs = []

    X_t = X.T.tocsr()  

    for params in param_grid:
    
        X_weighted = (X_t * params['alpha']).tocsr()

        model = AlternatingLeastSquares(
            factors=params['factors'],
            regularization=params['regularization'],
            iterations=20,
            random_state=42
        )
        model.fit(X_weighted)

        def _recommend(app_id, k=10, _model=model):
            game_idx = game_mapper[app_id]
            ids, scores = _model.similar_items(game_idx, N=k + 1)
            return [game_inv_mapper[i] for i in ids if i != game_idx][:k]

        metrics = evaluate_model(_recommend, train_df, test_positive)
        all_runs.append({**params, **metrics})
        print(f'{params} -> precision@k={metrics["precision@k"]:.5f}')

        if metrics['precision@k'] > best_score:
            best_score = metrics['precision@k']
            best_params = params
            best_model = model

    print(f'\nBest ALS params: {best_params} (precision@k={best_score:.5f})')

    def best_recommend(app_id, k=10):
        game_idx = game_mapper[app_id]
        ids, scores = best_model.similar_items(game_idx, N=k + 1)
        return [game_inv_mapper[i] for i in ids if i != game_idx][:k]

    return best_recommend, best_model, best_params, pd.DataFrame(all_runs)


#just returns top 10 games
# def make_popularity(train_df):
#     popularity_ranking = (
#         train_df[train_df['is_recommended'] == True]
#         .groupby('app_id')
#         .size()
#         .sort_values(ascending=False)
#         .index
#         .tolist()
#     )

#     def _recommend(app_id, k=10):
#         recs = [aid for aid in popularity_ranking if aid != app_id]
#         return recs[:k]
#     return _recommend


#lightfm model
def make_lightfm(train_df, content_idx_map, game_features, test_positive):
    if not LIGHTFM_AVAILABLE:
        return None, None, None, None

    param_grid = [
        {'no_components': 20, 'learning_rate': 0.05, 'loss': 'warp'},
        {'no_components': 50, 'learning_rate': 0.05, 'loss': 'warp'},
        {'no_components': 50, 'learning_rate': 0.01, 'loss': 'warp'},
        {'no_components': 50, 'learning_rate': 0.05, 'loss': 'bpr'},
        {'no_components': 100, 'learning_rate': 0.05, 'loss': 'warp'},
    ]

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
    n_items = len(item_id_map)

    feature_matrix_rows = []
    for app_id in item_id_map.keys():
        row_idx = content_idx_map.get(app_id)
        if row_idx is not None:
            feature_matrix_rows.append(game_features.iloc[row_idx].values)
        else:
            feature_matrix_rows.append(np.zeros(game_features.shape[1]))
    item_features_matrix = lightfm_csr(np.array(feature_matrix_rows, dtype=np.float32))
    best_score = -1
    best_params = None
    best_model = None
    all_runs = []

    for params in param_grid:
        model = LightFM(
            loss=params['loss'],
            no_components=params['no_components'],
            learning_rate=params['learning_rate'],
            random_state=42
        )
        model.fit(interactions, item_features=item_features_matrix, epochs=20, num_threads=4)

        def _recommend(app_id, k=10, _model=model):
            item_idx = item_id_map[app_id]
            scores = _model.predict(
                user_ids=np.zeros(n_items, dtype=int),
                item_ids=np.arange(n_items),
                item_features=item_features_matrix
            )
            top_indices = np.argsort(-scores)
            return [inv_item_map[i] for i in top_indices if inv_item_map[i] != app_id][:k]

        metrics = evaluate_model(_recommend, train_df, test_positive)
        all_runs.append({**params, **metrics})
        print(f'{params} -> precision@k={metrics["precision@k"]:.5f}')

        if metrics['precision@k'] > best_score:
            best_score = metrics['precision@k']
            best_params = params
            best_model = model

    print(f'\nBest LightFM params: {best_params} (precision@k={best_score:.5f})')

    def best_recommend(app_id, k=10):
        item_idx = item_id_map[app_id]
        scores = best_model.predict(
            user_ids=np.zeros(n_items, dtype=int),
            item_ids=np.arange(n_items),
            item_features=item_features_matrix
        )
        top_indices = np.argsort(-scores)
        return [inv_item_map[i] for i in top_indices if inv_item_map[i] != app_id][:k]

    return best_recommend, best_model, best_params, pd.DataFrame(all_runs)

#  Evaluation metrics

def evaluate_model(recommend_fn, train_df, test_df, k=10, n_users_sample=500):
    
    users_to_test = test_df['user_id'].unique()
    if len(users_to_test) > n_users_sample:
        users_to_test = np.random.choice(users_to_test, n_users_sample, replace=False)

    precisions, recalls = [], []

    for user_id in users_to_test:
        true_games = set(test_df.loc[test_df['user_id'] == user_id, 'app_id'])
        if not true_games:
            continue

        train_games = train_df.loc[
            (train_df['user_id'] == user_id) & (train_df['is_recommended'] == True),
            'app_id'
        ]
        if train_games.empty:
            continue

        seed_game = train_games.iloc[0]

        try:
            recommended = set(recommend_fn(seed_game, k=k))
        except KeyError:
            continue

        hits = recommended & true_games
        precisions.append(len(hits) / k)
        recalls.append(len(hits) / len(true_games))

    return {
        'precision@k': np.mean(precisions) if precisions else 0,
        'recall@k': np.mean(recalls) if recalls else 0,
        'n_users_evaluated': len(precisions)
    }



#Main

def main():
    #import data
    games_df, games_metadata_df, interactions_df = load_data()
    games_clean = clean_games(games_df)
    train_df, test_df, test_positive = split_data(interactions_df)

    #build csr matrix
    X, user_mapper, game_mapper, user_inv_mapper, game_inv_mapper, app_ids = \
        build_interaction_matrix(interactions_df, train_df)

    games_lookup = games_df[['app_id', 'title']].copy()

    # quick sanity check of KNN + fuzzy title match, mirrors original notebook
    sample_app_id = app_ids[0]
    print(game_finder('witcher wild hunt', games_lookup))

    #cosin sim method
    cosine_sim = cosine_similarity(X, X)
    print(f'Dimensions of cosine similarity matrix (train only): {cosine_sim.shape}')

    #build content based matrix
    content_df, game_features, content_idx_map, content_inv_map = \
        build_content_features(games_clean, games_metadata_df)

    

    # ---- build all model recommend_fns ----
    models = {}
    models['collaborative_knn'] = make_collaborative_knn(X, game_mapper, game_inv_mapper)
    models['collaborative_cosine'] = make_collaborative_cosine(cosine_sim, game_mapper, game_inv_mapper)
    models['content_based'] = make_content_based(game_features, content_idx_map, content_inv_map)
   
    als_recommend, als_model, als_best_params, als_runs = make_als(
        X, game_mapper, game_inv_mapper, train_df, test_positive
    )
    models['als'] = als_recommend

    lightfm_recommend, lightfm_model, lightfm_best_params, lightfm_runs = make_lightfm(
        train_df, content_idx_map, game_features, test_positive
    )
    if lightfm_recommend is not None:
        models['lightfm'] = lightfm_recommend

    # ---- evaluate all models with the same harness ----
    results = {}
    for name, recommend_fn in models.items():
        print(f'Evaluating {name}...')
        results[name] = evaluate_model(recommend_fn, train_df, test_positive)

    print('\n=== Results ===')
    results_df = pd.DataFrame(results).T.sort_values('precision@k', ascending=False)
    print(results_df)

    # save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_df.to_csv(f'results/model_comparison_{timestamp}.csv')

    best_model_name = results_df['precision@k'].idxmax()
    with open(f'results/best_model_{timestamp}.json', 'w') as f:
        json.dump({
            'best_model': best_model_name,
            'metrics': results_df.loc[best_model_name].to_dict(),
            'all_results': results_df.to_dict()
        }, f, indent=2)

    print(f"\nBest model: {best_model_name}")
    print(f"Results saved to results/model_comparison_{timestamp}.csv")
    print(f"Best model summary saved to results/best_model_{timestamp}.json")

    return results_df


if __name__ == '__main__':
    main()