import os
import re
import requests

STEAM_API_KEY = os.environ.get('STEAM_API_KEY')
STEAM_API_BASE = 'https://api.steampowered.com'


class SteamProfileError(Exception):
    pass


def _extract_identifier(profile_input):
    '''
    Accepts a raw SteamID64, a full profile URL, or a vanity name, and
    returns (kind, value) where kind is 'steamid' or 'vanity'.
    '''
    profile_input = profile_input.strip()

    # full URL like https://steamcommunity.com/id/somename or /profiles/7656119...
    match = re.search(r'steamcommunity\.com/(id|profiles)/([^/]+)', profile_input)
    if match:
        url_type, value = match.groups()
        return ('steamid' if url_type == 'profiles' else 'vanity', value)

    # raw 17-digit SteamID64
    if re.fullmatch(r'\d{17}', profile_input):
        return ('steamid', profile_input)

    # otherwise assume it's a vanity name typed directly
    return ('vanity', profile_input)


def resolve_to_steamid(profile_input):
    '''Turns any accepted input format into a numeric SteamID64.'''
    if not STEAM_API_KEY:
        raise SteamProfileError('STEAM_API_KEY environment variable is not set.')

    kind, value = _extract_identifier(profile_input)

    if kind == 'steamid':
        return value

    # vanity name -> needs a lookup call
    resp = requests.get(
        f'{STEAM_API_BASE}/ISteamUser/ResolveVanityURL/v0001/',
        params={'key': STEAM_API_KEY, 'vanityurl': value},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json().get('response', {})

    if data.get('success') != 1:
        raise SteamProfileError(f'Could not resolve Steam profile "{value}". Check the name/URL.')

    return data['steamid']


def get_owned_games(profile_input):
    '''
    Returns a list of dicts: [{'app_id': int, 'playtime_minutes': int}, ...]
    Raises SteamProfileError if the profile is private or doesn't exist.
    '''
    if not STEAM_API_KEY:
        raise SteamProfileError('STEAM_API_KEY environment variable is not set.')

    steam_id = resolve_to_steamid(profile_input)

    resp = requests.get(
        f'{STEAM_API_BASE}/IPlayerService/GetOwnedGames/v0001/',
        params={
            'key': STEAM_API_KEY,
            'steamid': steam_id,
            'include_appinfo': False,
            'include_played_free_games': True,
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json().get('response', {})

    if 'games' not in data:
        # empty response usually means the profile/game list is private
        raise SteamProfileError(
            'This Steam profile\'s game list is private or empty. '
            'The profile owner needs to set "Game details" to Public in their Steam privacy settings.'
        )

    return [
        {'app_id': g['appid'], 'playtime_minutes': g.get('playtime_forever', 0)}
        for g in data['games']
    ]
