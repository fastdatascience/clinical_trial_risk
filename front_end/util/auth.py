from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from six import iteritems, add_metaclass


@add_metaclass(ABCMeta)
class Auth(object):
    def __init__(self, app, authorization_hook=None, _overwrite_index=True, routes_pathname_prefix=None):
        self.app = app
        self._index_view_name = routes_pathname_prefix if routes_pathname_prefix != None else app.config['routes_pathname_prefix']
        if _overwrite_index:
            self._overwrite_index()
            self._protect_views()
        self._index_view_name = routes_pathname_prefix if routes_pathname_prefix != None else app.config['routes_pathname_prefix']
        self._auth_hooks = [authorization_hook] if authorization_hook else []

    def _overwrite_index(self):
        original_index = self.app.server.view_functions[self._index_view_name]

        self.app.server.view_functions[self._index_view_name] = \
            self.index_auth_wrapper(original_index)

    def _protect_views(self):
        # TODO - allow users to white list in case they add their own views
        for view_name, view_method in iteritems(
                self.app.server.view_functions):
            if view_name != self._index_view_name:
                self.app.server.view_functions[view_name] = \
                    self.auth_wrapper(view_method)

    def is_authorized_hook(self, func):
        self._auth_hooks.append(func)
        return func

    @abstractmethod
    def is_authorized(self):
        pass

    @abstractmethod
    def auth_wrapper(self, f):
        pass

    @abstractmethod
    def index_auth_wrapper(self, f):
        pass

    @abstractmethod
    def login_request(self):
        pass
