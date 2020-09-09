#!/use/bin/env python
"""
This class runs a simple WSGI server that receives GET requests containing a path
to an audio file to use as a sound target for synthesizer sound matching. It returns
the parameter settings as JSON.
"""

import os
import json
from copy import copy
from wsgiref.simple_server import make_server
from urllib.parse import parse_qs

import spiegelib as spgl


class SoundMatchHTTPServer():
    """
    Args:
        sound_matcher (:class:`~spiegelib.core.SoundMatch`): SountMatch object to use
        address (str, optional): address to run server at. Defaults to localhost
        port (int, optional): port to run server at. Defaults to 8000
    """

    def __init__(self, sound_matcher, address="localhost", port=8000):
        """
        Constructor
        """

        self.port = port
        self.address = address
        self.sound_matcher = sound_matcher


    def start(self):
        """
        Begin server
        """

        with make_server(self.address, self.port, self) as httpd:
            print("Server started. Serving at %s:%s" % (self.address, self.port))
            httpd.serve_forever()


    def __call__(self, environ, start_response):
        """
        This gets called any time there is a new request made to the server
        """

        method = environ['REQUEST_METHOD']
        if method == 'GET':
            return self.get(environ, start_response)

        status = '500 Internal Server Error'
        start_response(status, [])
        return []


    def get(self, environ, start_response):
        """
        Process a GET request
        """

        path = environ['PATH_INFO']

        if path == "/sound_match":
            query = parse_qs(environ['QUERY_STRING'])
            response_body = self.try_sound_match(query, start_response)

        else:
            response_body = b''
            start_response(
                '404 Not Found',
                SoundMatchHTTPServer.get_headers(response_body, 'text/html')
            )

        return [response_body]


    @staticmethod
    def get_headers(response_body, mime_type):
        """
        Return headers for HTML
        """

        response_headers = [
            ('Content-Type', mime_type),
            ('Content-Length', str(len(response_body)))
        ]

        return response_headers


    def try_sound_match(self, query, start_response):
        """
        Trys to run a sound match given a query string

        Args:
            query (dict): Query string, which should contain a link to a target
                audio file
        """

        target = query.get('target', None)
        response_body = b''

        if target is None:
            response_body = b'Query string was incorrect. Missing target parameter.'
            start_response(
                '404 Not Found',
                SoundMatchHTTPServer.get_headers(response_body, 'text/html')
            )
            return response_body

        # Load audio file to use as sound target
        target = os.path.abspath(target[0])
        audio = spgl.AudioBuffer()

        try:
            audio.load(target)
        except FileNotFoundError:
            response_body = b'Unable to load audio target file'
            start_response(
                '500 Internal Server Error',
                SoundMatchHTTPServer.get_headers(response_body, 'text/html')
            )
            return response_body


        # Attempt to get parameter settings, either just as parameters
        # from a parameter match -- or if a synthesizer is cofigured for
        # this sound matcher then run the full sound match and get the patch
        params = []
        try:
            if self.sound_matcher.synth is None:
                params = self.sound_matcher.match_parameters(audio, expand=True)
            else:
                _ = self.sound_matcher.match(audio)
                params = copy(self.sound_matcher.get_patch())
                self.sound_matcher.patch = None

        except Exception as error:
            response_body = 'Sound matching failed. %s' % str(error)
            response_body = bytes(response_body, 'utf-8')
            start_response(
                '500 Internal Server Error',
                SoundMatchHTTPServer.get_headers(response_body, 'text/html')
            )
            return response_body

        # Successfully got a sound matched patch setting. Return as JSON
        params = {'patch': params}
        params = json.dumps(params)
        response_body = bytes(params, 'utf-8')
        start_response('200 Okay', SoundMatchHTTPServer.get_headers(response_body,
                                                                    'application/json'))
        return response_body
