#!/use/bin/env python
"""
"""


from wsgiref.simple_server import make_server


# pylint: disable=too-many-instance-attributes
class SoundMatchServer():
    """
    """

    # pylint: disable=too-many-arguments
    def __init__(self, sound_match):
        """
        Constructor
        """

        self.port = 8000
        self.address = "localhost"
        self.sound_match = sound_match


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
        print(path)
        response_body = b''
        start_response('404 Not Found', SoundMatchServer.get_headers(response_body, 'text/html'))

        return [response_body]


    def post(self, environ, start_response):
        """
        Process a POST request
        """

        # the environment variable CONTENT_LENGTH may be empty or missing
        path = environ['PATH_INFO']
        print(path)
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
