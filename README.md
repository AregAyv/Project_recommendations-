# Steam Game Recommender

Steam lists tens of thousands of games, and finding the next one worth playing is hard. This project recommends similar games from a single title you already like, or from your own Steam library, and lets you choose which of 7 recommendation models produces the results.

## Google cloude link
https://game-recommender-699539648647.europe-west1.run.app/


**Stack:** Python, scikit-learn, implicit, LightFM, FastAPI, Streamlit, Docker, Google Cloud Run

## Dataset

[Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) (Kaggle, Anton Kozyriev):

- about 50,000 games with price, release date, rating, and tags
- user reviews, each with an `is_recommended` flag
- about 37,600 games have review interactions, and the user-game matrix is **99.99% sparse**

The dataset is downloaded automatically on the first run through `kagglehub`, so nothing needs to be downloaded by hand, and no data files are stored in the repository.

The exploratory analysis (data quality, distributions, sparsity, correlations, and the reasoning behind each engineered feature) is in `data_eda.ipynb`.

## Approach

**Feedback signal.** A positive review (`is_recommended = True`) counts as a "like". The models see only these likes (implicit feedback).

**Content features.** The 100 most frequent tags (one-hot), release decade (one-hot), and price tier (free, 0-5, 5-10, 10-20, 20-30, 30-60, 60+). Features are binary, so no scaling is applied.

**Models:**

| Model | Idea |
|---|---|
| `collaborative_knn` | Games liked by the same players are similar (cosine kNN on the user-game matrix) |
| `collaborative_cosine` | Same similarity computed from a full pairwise matrix during experiments. The API serves it through the same fitted index as `collaborative_knn`, which gives identical results without shipping a very large matrix |
| `content_based` | Games with similar tags, price tier, and decade are similar (cosine kNN on metadata). Works for games with no reviews |
| `als` | Alternating Least Squares matrix factorization with confidence weighting (`implicit`) |
| `lightfm` | Hybrid model learning from interactions and game metadata together |
| `popularity` | Baseline: the most-liked games overall, ignoring the input game |
| `hybrid` | Weighted rank fusion over several of the models above |

**Hybrid.** Each selected model returns a pool of candidates (3x the requested number). Each position earns points, scaled by the model's weight, and the points are summed per game. A game that several models rank highly rises to the top. The default weights are `collaborative_knn` 1.0, `content_based` 1.0, `als` 1.5, and `popularity` 0.5.

**Tuning.** ALS was tuned with grid search over factors (20, 50, 100), regularization (0.01, 0.1), and confidence scaling (1, 15, 40). The final model uses factors=20, regularization=0.01, alpha=15. LightFM uses 50 components, learning rate 0.05, and WARP loss. The chosen values are passed to the final models so they train quickly.

## Evaluation

All models are evaluated with the same procedure:

1. Interactions are split by date, with the first 80% for training and the last 20% for testing.
2. For each of 500 randomly sampled users, the first game they liked during training is the "seed".
3. Each model recommends the top 10 games for that seed.
4. The recommendations are compared with the games the user liked during the test period.

| Metric | What it measures |
|---|---|
| precision@k / recall@k | Share of recommendations that were hits / share of the user's later likes that were found |
| MRR | How high the first correct recommendation ranks |
| MAP@k | Rewards hits that appear early in the list |
| Coverage | Share of the catalog that is ever recommended (exposes models that only suggest blockbusters) |
| Diversity | How different the games within one list are from each other |
| Novelty | How niche, rather than mainstream, the recommended games are |

### Results

                      precision@k  recall@k       mrr     map@k  coverage  diversity    novelty  n_users_evaluated
collaborative_cosine     0.010031  0.073877  0.033999  0.024274  0.037773   0.616020  11.572354              319.0
collaborative_knn        0.009467  0.061443  0.037199  0.028185  0.037933   0.634219  11.104800              338.0
popularity               0.007599  0.051798  0.031483  0.025079  0.000293   0.624316   6.305300              329.0
hybrid                   0.002187  0.012865  0.009097  0.004189  0.050959   0.735957  14.685920              320.0
lightfm                  0.001466  0.007769  0.002182  0.001194  0.036601   0.555889  15.937859              341.0
als                      0.001246  0.012461  0.004599  0.004599  0.054822   0.754002  16.103244              321.0
content_based            0.001201  0.006607  0.005255  0.002703  0.042381   0.173184  14.085724              333.0


## Project structure

```
Final_project/
├── data_eda.ipynb               # exploratory data analysis
├── main.py                      # builds features, trains and evaluates all models
├── freeze_models.py             # trains final models and saves them to model_artifacts/
├── app.py                       # FastAPI backend
├── steam_client.py              # Steam Web API wrapper
├── streamlit_app.py             # Streamlit frontend
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .dockerignore 
├── model_artifacts/             # generated by freeze_models.py(not on github)
└── results/                     # generated by main.py (metrics per run)(not on github)
```

## Getting started

### 1. Install

Python 3.11 is required (`lightfm` does not build on Python 3.14).

```bash
python3 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
```

### 2. Train and evaluate

```bash
python3 main.py
```

Downloads the data, trains all models, prints the comparison table, and saves it to `results/`.

### 3. Save the final models

```bash
python3 freeze_models.py
```

Writes `model_artifacts/model_artifacts.joblib`, which the API loads at startup. The API never trains anything itself.

### 4. Run the API

```bash
export STEAM_API_KEY=your_key_here
uvicorn app:app --reload --port 8080
```

Interactive docs: http://localhost:8080/docs

### 5. Run the frontend (it will outamaticly also run backend)

```bash
streamlit run streamlit_app.py
```

By default it calls `http://localhost:8080`. Set `API_BASE_URL` to point at another backend.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Status check; lists the available models |
| `GET /models` | Lists the available models |
| `GET /recommend` | Games similar to an `app_id`, using the chosen `model` |
| `GET /recommend_from_profile` | Recommendations seeded by the most-played game in a Steam profile |

Example:

```bash
curl "http://localhost:8080/recommend?app_id=413150&model=hybrid&k=10"
```

For `/recommend_from_profile`, the Steam profile's **Game details** privacy setting must be **Public**. The profile can be given as a full URL, a vanity name, or a SteamID64. If a model has no data for the most-played game, the next most-played game is tried.

**Input validation.** Requests are validated with FastAPI and Pydantic: `app_id` must be an integer, `model` must be one of the available models, `k` must be between 1 and 50, and responses follow typed Pydantic schemas.

### Finding a game's app_id

- **Steam store URL:** the number is in the URL, for example `store.steampowered.com/app/413150/Stardew_Valley/` gives `413150`.
- **[SteamDB](https://steamdb.info):** search by name.

## Docker

Build and run the image locally:

```bash
docker build -t game-recommender .
docker run -p 8080:8080 -e STEAM_API_KEY=your_key_here game-recommender
```

Then open http://localhost:8080. Do not put spaces around the `=` in `-e STEAM_API_KEY=...`. The key is passed at run time and is never stored in the image.

`model_artifacts/` must exist before building, so run `python3 freeze_models.py` first.

## Deployment

The app is deployed on Google Cloud Run: 

#You can deploy your version by following steps under
```bash
gcloud run deploy game-recommender \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --set-env-vars STEAM_API_KEY=your_key_here
```

The first request after a quiet period can take a while because Cloud Run starts the container and loads the models.

## Limitations

- Each user is evaluated from one seed game, on a random sample of 500 users. Absolute precision values are small, which is normal for this kind of evaluation, and the comparison between models matters more than any single number.
- Users who first appear late in the data have no training history, so collaborative models cannot be evaluated for them.
- Collaborative models cannot recommend games that have no reviews. `content_based` covers that case.
- `lightfm` can be hard to install on newer Python versions. This project was built and tested on Python 3.11.
