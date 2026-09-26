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
from sklearn.model_selection import train_test_split

from rapidfuzz import process

from implicit.als import AlternatingLeastSquares



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


# Train/test split
# 80/20
def split_data(interactions_df, random_state=42):
    train_df, test_df = train_test_split(interactions_df, test_size=0.2, random_state=random_state)
    train_df = train_df.copy()
    test_df = test_df.copy()
    test_positive = test_df[test_df['is_recommended'] == True]
    return train_df, test_df, test_positive



#make csr matrix
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


#Content-based df

def build_content_features(games_clean, games_metadata_df):
    content_df = games_clean.merge(
        games_metadata_df[['app_id', 'tags']], on='app_id', how='left'
    )
    content_df['tags'] = content_df['tags'].apply(lambda x: x if isinstance(x, list) else [])

    tag_counts = Counter(t for tags in content_df['tags'] for t in tags)
    print(f'There are {len(tag_counts)} unique tags.')
    print('Top 10 tags:', tag_counts.most_common(10))

    top_tags = [t for t, _ in tag_counts.most_common(100)]
    tag_columns = pd.DataFrame(
        {t: content_df['tags'].apply(lambda x: int(t in x)) for t in top_tags},
        index=content_df.index)


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
    [tag_columns, decade_dummies, price_dummies],
    axis=1).fillna(0)

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
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(X)

    def _recommend(app_id, k=10):
        game_idx = game_mapper[app_id]
        distances, indices = knn_model.kneighbors(X[game_idx], n_neighbors=k + 1)
        return [game_inv_mapper[i] for i in indices[0] if i != game_idx][:k]

    return _recommend


def make_collaborative_cosine(cosine_sim, game_mapper, game_inv_mapper):
    def _recommend(app_id, k=10):
        game_ind = game_mapper[app_id]
        sim_scores = list(enumerate(cosine_sim[game_ind]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k + 1]
        return [game_inv_mapper[i] for i, _ in sim_scores]
    return _recommend


#Knn but with content based data
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



#als model (AlternatingLeastSquares)
def make_als(X, game_mapper, game_inv_mapper, factors=20, regularization=0.01, alpha=15, iterations=20):
   
    als_model = AlternatingLeastSquares(
        factors=factors, regularization=regularization, iterations=iterations, random_state=42
    )

    X_t = X.T.tocsr()  
    X_weighted = (X_t * alpha).tocsr() 
    als_model.fit(X_weighted)

    def _recommend(app_id, k=10):
        game_idx = game_mapper[app_id]
        ids, scores = als_model.similar_items(game_idx, N=k + 1)
        return [game_inv_mapper[i] for i in ids if i != game_idx][:k]

    return _recommend, als_model




#lightfm model
def make_lightfm(train_df, content_idx_map, game_features, no_components=50, learning_rate=0.05, loss='warp', epochs=20):
  
    if not LIGHTFM_AVAILABLE:
        return None, None

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

    lightfm_model = LightFM(loss=loss, no_components=no_components, learning_rate=learning_rate, random_state=42)
    lightfm_model.fit(interactions, item_features=item_features_matrix, epochs=epochs, num_threads=4)

    _, item_embeddings = lightfm_model.get_item_representations(features=item_features_matrix)

    def _recommend(app_id, k=10):
        item_idx = item_id_map[app_id]
        seed_vec = item_embeddings[item_idx]
        scores = item_embeddings @ seed_vec
        top_indices = np.argsort(-scores)
        return [inv_item_map[i] for i in top_indices if inv_item_map[i] != app_id][:k]

    return _recommend, lightfm_model


def make_popularity(train_df):
    popularity_ranking = (
        train_df[train_df['is_recommended'] == True]
        .groupby('app_id')
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    def _recommend(app_id, k=10):
        recs = [aid for aid in popularity_ranking if aid != app_id]
        return recs[:k]
    return _recommend

#  Evaluation metrics
def reciprocal_rank(recommended_list, true_games):
    for i, item in enumerate(recommended_list):
        if item in true_games:
            return 1.0 / (i + 1)
    return 0.0


def average_precision_at_k(recommended_list, true_games, k=10):
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended_list[:k]):
        if item in true_games:
            hits += 1
            sum_precisions += hits / (i + 1)
    if hits == 0:
        return 0.0
    return sum_precisions / min(len(true_games), k)


def diversity_at_k(recommended_list, feature_lookup):
    vectors = [feature_lookup[aid] for aid in recommended_list if aid in feature_lookup]
    if len(vectors) < 2:
        return 0.0

    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_similarity(vectors[i], vectors[j])[0][0]
            sims.append(sim)

    avg_similarity = np.mean(sims)
    return 1 - avg_similarity


def novelty_at_k(recommended_list, item_popularity, n_users):
    novelties = []
    for item in recommended_list:
        pop = item_popularity.get(item, 0)
        if pop == 0:
            continue
        prob = pop / n_users
        novelties.append(-np.log2(prob))
    return np.mean(novelties) if novelties else 0.0


def build_evaluation_helpers(train_df, game_features, content_idx_map):
    feature_lookup = {}
    feature_values = game_features.astype(np.float32).values
    for app_id, row_idx in content_idx_map.items():
        feature_lookup[app_id] = feature_values[row_idx].reshape(1, -1)

    item_popularity = (
        train_df[train_df['is_recommended'] == True]
        .groupby('app_id')
        .size()
        .to_dict()
    )

    n_users = train_df['user_id'].nunique()

    return feature_lookup, item_popularity, n_users


def evaluate_model(recommend_fn, train_df, test_df, k=10, n_users_sample=500,
                    feature_lookup=None, item_popularity=None, n_users=None):
    users_to_test = test_df['user_id'].unique()
    if len(users_to_test) > n_users_sample:
        users_to_test = np.random.choice(users_to_test, n_users_sample, replace=False)

    precisions, recalls, rrs, aps, diversities, novelties = [], [], [], [], [], []
    all_recommended_games = set()

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
            recommended_list = recommend_fn(seed_game, k=k)
        except KeyError:
            continue

        recommended = set(recommended_list)
        all_recommended_games.update(recommended)

        hits = recommended & true_games
        precisions.append(len(hits) / k)
        recalls.append(len(hits) / len(true_games))
        rrs.append(reciprocal_rank(recommended_list, true_games))
        aps.append(average_precision_at_k(recommended_list, true_games, k=k))

        if feature_lookup is not None:
            diversities.append(diversity_at_k(recommended_list, feature_lookup))
        if item_popularity is not None and n_users is not None:
            novelties.append(novelty_at_k(recommended_list, item_popularity, n_users))

    n_total_games = train_df['app_id'].nunique()
    coverage = len(all_recommended_games) / n_total_games if n_total_games > 0 else 0

    return {
        'precision@k': np.mean(precisions) if precisions else 0,
        'recall@k': np.mean(recalls) if recalls else 0,
        'mrr': np.mean(rrs) if rrs else 0,
        'map@k': np.mean(aps) if aps else 0,
        'coverage': coverage,
        'diversity': np.mean(diversities) if diversities else None,
        'novelty': np.mean(novelties) if novelties else None,
        'n_users_evaluated': len(precisions)
    }

      

def make_hybrid(model_recommend_fns, weights=None, pool_multiplier=3):
    if weights is None:
        weights = {name: 1.0 for name in model_recommend_fns}

    def _recommend(app_id, k=10):
        scores = {}
        pool_size = k * pool_multiplier

        for name, recommend_fn in model_recommend_fns.items():
            weight = weights.get(name, 1.0)
            try:
                candidates = recommend_fn(app_id, k=pool_size)
            except KeyError:
                continue
                

            for rank, candidate_id in enumerate(candidates):
                points = (pool_size - rank) * weight
                scores[candidate_id] = scores.get(candidate_id, 0) + points

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in ranked[:k]]

    return _recommend


#Main
def main():
    #import data
    games_df, games_metadata_df, interactions_df = load_data()
    games_clean = clean_games(games_df)
    train_df, test_df, test_positive = split_data(interactions_df)

    #build csr matrix
    X, user_mapper, game_mapper, user_inv_mapper, game_inv_mapper, app_ids = \
        build_interaction_matrix(interactions_df, train_df)


        #build content based matrix
    content_df, game_features, content_idx_map, content_inv_map = \
        build_content_features(games_clean, games_metadata_df)

    #cosine sim method (full pairwise matrix -- heavier on memory than collaborative_knn)
    cosine_sim = cosine_similarity(X, X)
    print(f'Dimensions of cosine similarity matrix (train only): {cosine_sim.shape}')

    # ---- build all model recommend_fns ----
    models = {}
    models['collaborative_knn'] = make_collaborative_knn(X, game_mapper, game_inv_mapper)
    models['collaborative_cosine'] = make_collaborative_cosine(cosine_sim, game_mapper, game_inv_mapper)
    models['content_based'] = make_content_based(game_features, content_idx_map, content_inv_map)

    als_recommend, als_model = make_als(X, game_mapper, game_inv_mapper)
    models['als'] = als_recommend

    models['popularity'] = make_popularity(train_df)

    lightfm_recommend, lightfm_model = make_lightfm(train_df, content_idx_map, game_features)
    if lightfm_recommend is not None:
        models['lightfm'] = lightfm_recommend


        lightfm_recommend, lightfm_model = make_lightfm(train_df, content_idx_map, game_features)
    if lightfm_recommend is not None:
        models['lightfm'] = lightfm_recommend

    # hybrid model
    hybrid_selection = {
        'collaborative_knn': 1.0,
        'collaborative_cosine': 1.0,
        'content_based': 1.0,
        'als': 1.5,
        'popularity': 0.5,
        'lightfm': 1.0,
    }
    hybrid_models = {name: models[name] for name in hybrid_selection if name in models}
    hybrid_weights = {name: w for name, w in hybrid_selection.items() if name in models}

    models['hybrid'] = make_hybrid(hybrid_models, weights=hybrid_weights)

    # ---- evaluate all models with the same harness ----
    feature_lookup, item_popularity, n_users = build_evaluation_helpers(
    train_df, game_features, content_idx_map
)

    results = {}
    for name, recommend_fn in models.items():
        print(f'Evaluating {name}...')
        results[name] = evaluate_model(
            recommend_fn, train_df, test_positive,
            feature_lookup=feature_lookup,
            item_popularity=item_popularity,
            n_users=n_users
        )

    print('\n=== Results ===')
    results_df = pd.DataFrame(results).T.sort_values('precision@k', ascending=False)
    print(results_df)

    return 


if __name__ == '__main__':
    main()