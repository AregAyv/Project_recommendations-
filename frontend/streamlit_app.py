import os
import requests
import streamlit as st

# API_BASE_URL = 'https://game-recommender-api-xxxx.us-central1.run.app'
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8080')

st.set_page_config(page_title='Game Recommender', page_icon='🎮', layout='centered')
st.title('🎮 Game Recommender')


@st.cache_data(ttl=300)
def get_available_models():
    resp = requests.get(f'{API_BASE_URL}/health', timeout=10)
    resp.raise_for_status()
    return resp.json()['available_models']

try:
    available_models = get_available_models()
except requests.exceptions.RequestException as e:
    st.error(f'Could not reach the API at {API_BASE_URL}. Is it running? ({e})')
    st.stop()

mode = st.radio('Recommend based on:', ['A specific game', 'My Steam profile'])

model = st.selectbox('Model', available_models, index=available_models.index('hybrid') if 'hybrid' in available_models else 0)
k = st.slider('Number of recommendations', min_value=1, max_value=50, value=10)

if mode == 'A specific game':
    app_id = st.number_input('Steam app_id', min_value=1, value=413150, step=1)

    if st.button('Get recommendations'):
        with st.spinner('Fetching recommendations...'):
            resp = requests.get(
                f'{API_BASE_URL}/recommend',
                params={'app_id': app_id, 'model': model, 'k': k},
                timeout=30,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.subheader(f'Because you played: {data["seed_title"]}')
            st.caption(f'Model used: {data["model_used"]}')
            for rec in data['recommendations']:
                st.write(f'- **{rec["title"]}**  (app_id: {rec["app_id"]})')
        else:
            st.error(resp.json().get('detail', 'Something went wrong.'))

else:
    steam_profile = st.text_input('Steam profile URL, vanity name, or SteamID64')

    if st.button('Get recommendations'):
        if not steam_profile:
            st.warning('Enter a Steam profile first.')
        else:
            with st.spinner('Fetching your library and recommendations...'):
                resp = requests.get(
                    f'{API_BASE_URL}/recommend_from_profile',
                    params={'steam_profile': steam_profile, 'model': model, 'k': k},
                    timeout=30,
                )
            if resp.status_code == 200:
                data = resp.json()
                st.subheader(f'Because you played: {data["seed_title"]}')
                st.caption(f'{data["seed_reason"]}  •  {data["games_owned_matched"]} owned games matched  •  model: {data["model_used"]}')
                for rec in data['recommendations']:
                    st.write(f'- **{rec["title"]}**  (app_id: {rec["app_id"]})')
            else:
                st.error(resp.json().get('detail', 'Something went wrong.'))

st.divider()
st.caption(f'Connected to API: {API_BASE_URL}')
