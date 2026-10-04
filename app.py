import os
from enum import Enum

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from steam_client import get_owned_games, SteamProfileError

ARTIFACTS_PATH = os.environ.get('ARTIFACTS_PATH', 'model_artifacts/model_artifacts.joblib')

app = FastAPI(
    title='Game Recommender API',
    description='Recommend similar Steam games using one of several models, or a weighted hybrid blend.',
    version='1.0.0',
)
#load artifacts
artifacts = joblib.load(ARTIFACTS_PATH)

game_mapper = artifacts['game_mapper']
game_inv_mapper = artifacts['game_inv_mapper']
content_idx_map = artifacts['content_idx_map']
content_inv_map = artifacts['content_inv_map']
knn_collab_model = artifacts['knn_collab_model']
knn_content_model = artifacts['knn_content_model']
als_model = artifacts['als_model']
popularity_ranking = artifacts['popularity_ranking']
games_lookup = artifacts['games_lookup']
available_models = artifacts['available_models']

lightfm_item_embeddings = artifacts.get('lightfm_item_embeddings')
lightfm_item_id_map = artifacts.get('lightfm_item_id_map')
lightfm_inv_item_map = artifacts.get('lightfm_inv_item_map')


class ModelName(str, Enum):
    collaborative_knn = 'collaborative_knn'
    collaborative_cosine = 'collaborative_cosine'
    content_based = 'content_based'
    als = 'als'
    popularity = 'popularity'
    lightfm = 'lightfm'
    hybrid = 'hybrid'


class Recommendation(BaseModel):
    app_id: int
    title: str


class RecommendResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    seed_app_id: int
    seed_title: str
    model_used: str
    recommendations: list[Recommendation]


class ProfileRecommendResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    steam_profile: str
    games_owned_matched: int
    seed_app_id: int
    seed_title: str
    seed_reason: str
    model_used: str
    recommendations: list[Recommendation]


#functions for each model
def recommend_collaborative_knn(app_id, k):
    game_idx = game_mapper[app_id]
    distances, indices = knn_collab_model.kneighbors(
        knn_collab_model._fit_X[game_idx], n_neighbors=k + 1
    )
    return [game_inv_mapper[i] for i in indices[0] if i != game_idx][:k]


def recommend_collaborative_cosine(app_id, k):
    return recommend_collaborative_knn(app_id, k)


def recommend_content_based(app_id, k):
    idx = content_idx_map[app_id]
    distances, indices = knn_content_model.kneighbors(
        knn_content_model._fit_X[idx], n_neighbors=k + 1
    )
    return [content_inv_map[i] for i in indices[0] if i != idx][:k]


def recommend_als(app_id, k):
    game_idx = game_mapper[app_id]
    ids, scores = als_model.similar_items(game_idx, N=k + 1)
    return [game_inv_mapper[i] for i in ids if i != game_idx][:k]


def recommend_popularity(app_id, k):
    recs = [aid for aid in popularity_ranking if aid != app_id]
    return recs[:k]


def recommend_lightfm(app_id, k):
    if lightfm_item_embeddings is None:
        raise HTTPException(status_code=404, detail='LightFM model is not available on this deployment.')
    item_idx = lightfm_item_id_map[app_id]
    seed_vec = lightfm_item_embeddings[item_idx]
    scores = lightfm_item_embeddings @ seed_vec
    top_indices = np.argsort(-scores)
    return [lightfm_inv_item_map[i] for i in top_indices if lightfm_inv_item_map[i] != app_id][:k]



DEFAULT_HYBRID_WEIGHTS = {
    'collaborative_knn': 1.0,
    'content_based': 1.0,
    'als': 1.5,
    'popularity': 0.5,
}


def recommend_hybrid(app_id, k, weights=None, pool_multiplier=3):
    if weights is None:
        weights = DEFAULT_HYBRID_WEIGHTS

    scores = {}
    pool_size = k * pool_multiplier

    for name, weight in weights.items():
        recommend_fn = MODEL_DISPATCH.get(name)
        if recommend_fn is None or name == 'hybrid':
            continue
        try:
            candidates = recommend_fn(app_id, pool_size)
        except KeyError:
            continue
        for rank, candidate_id in enumerate(candidates):
            points = (pool_size - rank) * weight
            scores[candidate_id] = scores.get(candidate_id, 0) + points

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [aid for aid, _ in ranked[:k]]



MODEL_DISPATCH = {
    'collaborative_knn': recommend_collaborative_knn,
    'collaborative_cosine': recommend_collaborative_cosine,
    'content_based': recommend_content_based,
    'als': recommend_als,
    'popularity': recommend_popularity,
    'lightfm': recommend_lightfm,
    'hybrid': recommend_hybrid,
}


@app.get('/health')
def health():
    return {'status': 'ok', 'available_models': available_models}


@app.get('/models')
def list_models():
    return {'available_models': available_models}


@app.get('/recommend', response_model=RecommendResponse)
def recommend(
    app_id: int = Query(..., description='Steam app_id of the game to base recommendations on'),
    model: ModelName = Query(..., description='Which model to use for recommendations'),
    k: int = Query(10, ge=1, le=50, description='Number of recommendations to return'),
):
    if model.value not in available_models:
        raise HTTPException(status_code=400, detail=f'Model "{model.value}" is not available on this deployment.')

    if app_id not in games_lookup:
        raise HTTPException(status_code=404, detail=f'app_id {app_id} not found.')

    recommend_fn = MODEL_DISPATCH[model.value]

    try:
        recommended_ids = recommend_fn(app_id, k)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f'app_id {app_id} was not present in this model\'s training data.'
        )

    return RecommendResponse(
        seed_app_id=app_id,
        seed_title=games_lookup.get(app_id, 'Unknown'),
        model_used=model.value,
        recommendations=[
            Recommendation(app_id=aid, title=games_lookup.get(aid, 'Unknown'))
            for aid in recommended_ids
        ],
    )


@app.get('/recommend_from_profile', response_model=ProfileRecommendResponse)
def recommend_from_profile(
    steam_profile: str = Query(
        ..., description='SteamID64, full profile URL, or vanity name (e.g. steamcommunity.com/id/yourname)'
    ),
    model: ModelName = Query(..., description='Which model to use for recommendations'),
    k: int = Query(10, ge=1, le=50, description='Number of recommendations to return'),
):
    try:
        owned_games = get_owned_games(steam_profile)
    except SteamProfileError as e:
        raise HTTPException(status_code=400, detail=str(e))

    known_owned = [g for g in owned_games if g['app_id'] in games_lookup]
    if not known_owned:
        raise HTTPException(
            status_code=404,
            detail='None of this profile\'s owned games are in our dataset.'
        )

    recommend_fn = MODEL_DISPATCH[model.value]

    # try candidates in order of playtime, skipping any the chosen model
    # doesn't actually have training data for, instead of failing outright
    # on just the single most-played game
    candidates_by_playtime = sorted(known_owned, key=lambda g: g['playtime_minutes'], reverse=True)

    seed_game = None
    recommended_ids = None
    for candidate in candidates_by_playtime:
        try:
            recommended_ids = recommend_fn(candidate['app_id'], k)
            seed_game = candidate
            break
        except KeyError:
            continue

    if seed_game is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f'None of this profile\'s {len(known_owned)} matched games have training '
                f'data for the "{model.value}" model. Try a different model.'
            )
        )

    seed_app_id = seed_game['app_id']
        

    owned_ids = {g['app_id'] for g in owned_games}
    recommended_ids = [aid for aid in recommended_ids if aid not in owned_ids]

    return ProfileRecommendResponse(
        steam_profile=steam_profile,
        games_owned_matched=len(known_owned),
        seed_app_id=seed_app_id,
        seed_title=games_lookup.get(seed_app_id, 'Unknown'),
        seed_reason=f'Most-played game in this profile with data available for this model ({seed_game["playtime_minutes"]} minutes)',
        model_used=model.value,
        recommendations=[
            Recommendation(app_id=aid, title=games_lookup.get(aid, 'Unknown'))
            for aid in recommended_ids
        ],
    )