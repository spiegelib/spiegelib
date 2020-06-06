#!/use/bin/env python
"""
"""

import os
import json
from copy import copy
from wsgiref.simple_server import make_server
from urllib.parse import parse_qs

import spiegelib as spgl


# pylint: disable=too-many-instance-attributes
class SoundMatchServer():
    """
    """

    # pylint: disable=too-many-arguments
    def __init__(self, sound_matcher):
        """
        Constructor
        """

        self.port = 8000
        self.address = "localhost"
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

        if method == 'POST':
            return self.post(environ, start_response)

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
                SoundMatchServer.get_headers(response_body, 'text/html')
            )

        return [response_body]


    def post(self, environ, start_response):
        """
        Process a POST request
        """

        # the environment variable CONTENT_LENGTH may be empty or missing
        path = environ['PATH_INFO']
        response_body = b''
        start_response('404 Not Found', SoundMatchServer.get_headers(response_body, 'text/html'))

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
            start_response(
                '404 Not Found',
                SoundMatchServer.get_headers(response_body, 'text/html')
            )
            return response_body

        target = os.path.abspath(target[0])
        audio = spgl.AudioBuffer()

        try:
            audio.load(target)
        except FileNotFoundError:
            start_response(
                '404 Not Found',
                SoundMatchServer.get_headers(response_body, 'text/html')
            )
            return response_body

        params = []
        try:
            _ = self.sound_matcher.match(audio)
            params = copy(self.sound_matcher.get_patch(skip_overridden=False))
            self.sound_matcher.patch = None
        except Exception as e:
            start_response(
                '500 Internal Server Error',
                SoundMatchServer.get_headers(response_body, 'text/html')
            )
            return response_body

        params = {'patch': params}
        params = json.dumps(params)
        response_body = bytes(params, 'utf-8')
        start_response('200 Okay', SoundMatchServer.get_headers(response_body, 'application/json'))
        return response_body
