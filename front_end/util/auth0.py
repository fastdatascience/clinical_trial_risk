import os

import flask
from authlib.integrations.requests_client import OAuth2Session
import traceback

from flask import url_for
from urllib.parse import urlencode, urljoin

from requests import request

from dash_auth0_oauth.auth import Auth

COOKIE_EXPIRY = 60 * 60 * 24 * 14
COOKIE_AUTH_USER_NAME = 'AUTH-USER'
COOKIE_AUTH_ACCESS_TOKEN = 'AUTH-TOKEN'

AUTH_STATE_KEY = 'auth_state'

CLIENT_ID = os.environ.get('AUTH0_AUTH_CLIENT_ID')
CLIENT_SECRET = os.environ.get('AUTH0_AUTH_CLIENT_SECRET')
LOGOUT_URL = os.environ.get('AUTH0_LOGOUT_URL')
AUTH_REDIRECT_URI = '/login/callback'

AUTH_FLASK_ROUTES = os.environ.get('AUTH_FLASK_ROUTES', "false")
if AUTH_FLASK_ROUTES == "true":
    AUTH_FLASK_ROUTES = True
if AUTH_FLASK_ROUTES == "false":
    AUTH_FLASK_ROUTES = False
else:
    print(
        f"warning: AUTH_FLASK_ROUTES is set to {AUTH_FLASK_ROUTES}. Must be 'true' or 'false', otherwise will raise this warning and be set to False.")
    AUTH_FLASK_ROUTES = False


class Auth0Auth(Auth):
    def __init__(self, app):
        Auth.__init__(self, app)
        app.server.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
        app.server.config['SESSION_TYPE'] = 'filesystem'

        @app.server.route('/login/callback')
        def callback():
            return self.login_callback()

        @app.server.route('/logout/')
        def logout():
            return self.logout()

    def is_authorized(self):
        user = flask.request.cookies.get(COOKIE_AUTH_USER_NAME)
        token = flask.request.cookies.get(COOKIE_AUTH_ACCESS_TOKEN)
        if not user or not token:
            return False
        return flask.session.get(user) == token

    def login_request(self):

        redirect_uri = urljoin(flask.request.base_url, AUTH_REDIRECT_URI)

        session = OAuth2Session(
            CLIENT_ID,
            CLIENT_SECRET,
            scope=os.environ.get('AUTH0_AUTH_SCOPE'),
            redirect_uri=redirect_uri
        )

        uri, state = session.create_authorization_url(
            os.environ.get('AUTH0_AUTH_URL'),
            audience=os.environ.get('AUTH0_API_AUDIENCE')
        )

        flask.session['REDIRECT_URL'] = flask.request.url
        flask.session[AUTH_STATE_KEY] = state
        flask.session.permanent = False

        return flask.redirect(uri, code=302)

    def auth_wrapper(self, f):
        def wrap(*args, **kwargs):
            if AUTH_FLASK_ROUTES:
                if not self.is_authorized():
                    return flask.Response(status=403)
            response = f(*args, **kwargs)
            return response

        return wrap

    def index_auth_wrapper(self, original_index):
        def wrap(*args, **kwargs):
            if self.is_authorized():
                return original_index(*args, **kwargs)
            else:
                return self.login_request()

        return wrap

    def login_callback(self):
        print("login callback")
        if 'error' in flask.request.args:
            if flask.request.args.get('error') == 'access_denied':
                return 'You denied access.'
            return 'Error encountered.'

        if 'code' not in flask.request.args and 'state' not in flask.request.args:
            return self.login_request()
        else:
            # user is successfully authenticated
            auth0 = self.__get_auth(state=flask.session[AUTH_STATE_KEY])
            try:
                token = auth0.fetch_token(
                    os.environ.get('AUTH0_AUTH_TOKEN_URI'),
                    client_secret=CLIENT_SECRET,
                    authorization_response=flask.request.url
                )
                print(token)
                # print(token['id_token'])
                print(token['access_token'])
            except Exception as e:
                print("exception")
                print (e)
                traceback.print_exc()
                return e.__dict__

            auth0 = self.__get_auth(token=token)
            resp = auth0.get(os.environ.get('AUTH0_AUTH_USER_INFO_URL'))
            print("response")
            print(resp.json())
            if resp.status_code == 200:
                print("this is back")
                user_data = resp.json()
                r = flask.redirect(flask.session['REDIRECT_URL'])
                r.set_cookie(COOKIE_AUTH_USER_NAME, user_data['name'], max_age=COOKIE_EXPIRY)
                r.set_cookie(COOKIE_AUTH_ACCESS_TOKEN, token['access_token'], max_age=COOKIE_EXPIRY)
                flask.session[user_data['name']] = token['access_token']
                return r

            return 'Could not fetch your information.'

    @staticmethod
    def __get_auth(state=None, token=None):
        if token:
            return OAuth2Session(CLIENT_ID, token=token)
        if state:
            return OAuth2Session(
                CLIENT_ID,
                state=state,
                redirect_uri=urljoin(flask.request.base_url, AUTH_REDIRECT_URI)
            )
        return OAuth2Session(
            CLIENT_ID,
            redirect_uri=urljoin(flask.request.base_url, AUTH_REDIRECT_URI),
        )

    @staticmethod
    def logout():

        # Clear session stored data
        flask.session.clear()

        # Redirect user to logout endpoint
        return_url = flask.request.host_url
        params = {'returnTo': return_url, 'client_id': CLIENT_ID}
        r = flask.redirect(LOGOUT_URL + '?' + urlencode(params))
        r.delete_cookie(COOKIE_AUTH_USER_NAME)
        r.delete_cookie(COOKIE_AUTH_ACCESS_TOKEN)

        return r
