import atexit
import os
import socket
import subprocess
import sys
import time

import requests
import streamlit as st

# must be the very first Streamlit command in the script
st.set_page_config(page_title='Game Recommender', page_icon='🎮', layout='centered')

# INTERNAL_API_PORT is the port the background API process listens on.
# In a combined image this must differ from Streamlit's own external port,
# so it defaults to 9000 there; locally (two separate folders/containers)
# it still defaults to 8080 to match the existing instructions.
API_PORT = int(os.environ.get('INTERNAL_API_PORT', 8080))
API_BASE_URL = os.environ.get('API_BASE_URL', f'http://localhost:{API_PORT}')


def find_project_root():
    '''
    Locates the folder containing app.py. Works whether streamlit_app.py sits
    next to app.py (combined image) or one folder below it (local frontend/ setup).
    '''
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, 'app.py')):
        return here
    parent = os.path.abspath(os.path.join(here, '..'))
    return parent


PROJECT_ROOT = find_project_root()
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uvicorn.log')


def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_api_if_needed():
    '''Starts uvicorn in the background if nothing is already listening on API_PORT.'''
    if is_port_open(API_PORT):
        return  # already running -- either we started it earlier, or it's running manually

    log = open(LOG_FILE, 'w')
    process = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'app:app', '--port', str(API_PORT)],
        cwd=PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    atexit.register(process.terminate)

    # wait until it's actually accepting connections (up to ~15 seconds)
    for _ in range(30):
        if is_port_open(API_PORT):
            return
        time.sleep(0.5)


# only auto-start a local API when API_BASE_URL still points at localhost --
# once deployed, API_BASE_URL is set to the real deployed backend URL, and
# app.py won't even be present in this container, so this is skipped entirely
is_local = 'localhost' in API_BASE_URL or '127.0.0.1' in API_BASE_URL

if is_local and 'api_started' not in st.session_state:
    with st.spinner('Starting backend API...'):
        start_api_if_needed()
    st.session_state['api_started'] = True


st.title('🎮 Game Recommender')

# ---- fetch available models once, from the live API ----
@st.cache_data(ttl=300)
def get_available_models():
    resp = requests.get(f'{API_BASE_URL}/health', timeout=10)
    resp.raise_for_status()
    return resp.json()['available_models']

try:
    available_models = get_available_models()
except requests.exceptions.RequestException as e:
    st.error(
        f'Could not reach the API at {API_BASE_URL}. '
        f'Check {LOG_FILE} for startup errors. ({e})'
    )
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