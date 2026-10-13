import atexit
import os
import socket
import subprocess
import sys
import time

import requests
import streamlit as st

st.set_page_config(page_title='Game Recommender', page_icon='🎮', layout='centered')

API_PORT = int(os.environ.get('INTERNAL_API_PORT', 8080))
API_BASE_URL = os.environ.get('API_BASE_URL', f'http://localhost:{API_PORT}')

def find_project_root():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, 'app.py')):
        return here
    return os.path.abspath(os.path.join(here, '..'))

PROJECT_ROOT = find_project_root()
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uvicorn.log')

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_api_if_needed():
    if is_port_open(API_PORT):
        return

    log = open(LOG_FILE, 'w')
    process = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'uvicorn',
            'app:app',
            '--host',
            '127.0.0.1',
            '--port',
            str(API_PORT)
        ],
        cwd=PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    atexit.register(process.terminate)

    for _ in range(120):
        if process.poll() is not None:
            log.close()
            return

        if is_port_open(API_PORT):
            return

        time.sleep(1)

    log.close()

is_local = 'localhost' in API_BASE_URL or '127.0.0.1' in API_BASE_URL

if is_local and 'api_started' not in st.session_state:
    with st.spinner('Starting backend API...'):
        start_api_if_needed()
    st.session_state['api_started'] = True

st.title('🎮 Game Recommender')

@st.cache_data(ttl=300)
def get_available_models():
    resp = requests.get(f'{API_BASE_URL}/health', timeout=10)
    resp.raise_for_status()
    return resp.json()['available_models']

try:
    available_models = get_available_models()
except requests.exceptions.RequestException as e:
    st.error(f'Could not reach the API at {API_BASE_URL}. Error: {e}')

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as log:
            st.code(log.read()[-10000:])

    st.stop()
except (KeyError, ValueError) as e:
    st.error(f'The API returned an unexpected health response: {e}')
    st.stop()

if not available_models:
    st.error('The API did not return any available recommendation models.')
    st.stop()

default_model_index = available_models.index('hybrid') if 'hybrid' in available_models else 0

mode = st.radio('Recommend based on:', ['A specific game', 'My Steam profile'])
model = st.selectbox('Model', available_models, index=default_model_index)
k = st.slider('Number of recommendations', min_value=1, max_value=50, value=10)

if mode == 'A specific game':
    app_id = st.number_input('Steam app_id', min_value=1, value=413150, step=1)

    if st.button('Get recommendations'):
        try:
            with st.spinner('Fetching recommendations...'):
                resp = requests.get(
                    f'{API_BASE_URL}/recommend',
                    params={'app_id': app_id, 'model': model, 'k': k},
                    timeout=120,
                )

            if resp.status_code == 200:
                data = resp.json()
                st.subheader(f'Because you played: {data["seed_title"]}')
                st.caption(f'Model used: {data["model_used"]}')

                for rec in data['recommendations']:
                    st.write(f'- **{rec["title"]}** (app_id: {rec["app_id"]})')
            else:
                try:
                    detail = resp.json().get('detail', 'Something went wrong.')
                except ValueError:
                    detail = resp.text
                st.error(detail)

        except requests.exceptions.RequestException as e:
            st.error(f'Recommendation request failed: {e}')

else:
    steam_profile = st.text_input('Steam profile URL, vanity name, or SteamID64')

    if st.button('Get recommendations'):
        if not steam_profile:
            st.warning('Enter a Steam profile first.')
        else:
            try:
                with st.spinner('Fetching your library and recommendations...'):
                    resp = requests.get(
                        f'{API_BASE_URL}/recommend_from_profile',
                        params={
                            'steam_profile': steam_profile,
                            'model': model,
                            'k': k
                        },
                        timeout=120,
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    st.subheader(f'Because you played: {data["seed_title"]}')
                    st.caption(
                        f'{data["seed_reason"]} • '
                        f'{data["games_owned_matched"]} owned games matched • '
                        f'model: {data["model_used"]}'
                    )

                    for rec in data['recommendations']:
                        st.write(f'- **{rec["title"]}** (app_id: {rec["app_id"]})')
                else:
                    try:
                        detail = resp.json().get('detail', 'Something went wrong.')
                    except ValueError:
                        detail = resp.text
                    st.error(detail)

            except requests.exceptions.RequestException as e:
                st.error(f'Recommendation request failed: {e}')

st.divider()
st.caption(f'Connected to API: {API_BASE_URL}')