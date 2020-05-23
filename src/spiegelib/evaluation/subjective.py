#!/use/bin/env python
"""
Basic subjective audio evaluation class
"""

import os
import json
import tempfile
from wsgiref.simple_server import make_server
from urllib import parse
import scipy.io.wavfile
from spiegelib.evaluation import EvaluationBase

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import beaqlejs


# pylint: disable=too-many-instance-attributes
# Need more instance attributes here
class Subjective(EvaluationBase):
    """
    A WSGI Application for basic subjective evaluation similarity tests between
    targets and estimations (or any other sound file for that matter)

    :param targets: A list of target AudioBuffers
    :type targets:
    """

    def __init__(self, targets, estimations, max_tests=5, output_dir='./'):
        """
        Constructor
        """

        super().__init__(targets, estimations)

        self.test_data = {}
        self.port = 8000
        self.address = "localhost"
        self.max_tests = max_tests
        self.output_dir = os.getcwd() if output_dir == './' else output_dir

        self.setup_statics()
        self.setup_test()
        self.create_config()


    def evaluate(self):
        """
        Run the web server that hosts that evaluation
        """

        with make_server(self.address, self.port, self) as httpd:
            print("serving at %s:%s" % (self.address, self.port))
            httpd.serve_forever()


    def evaluate_target(self, target, predictions):
        pass


    def verify_input_list(self, input_list):
        """
        Override verify input list for checking audio lists
        """
        EvaluationBase.verify_audio_input_list(input_list)


    def setup_test(self):
        """
        Setup the test dictionary and also save all audio files into an audio
        file dictionary keyed on their path that will be requested.

        :meta private:
        """

        self.audio_files = {}
        self.audio_folder = 'audio'
        self.test_data['targets'] = []
        self.test_data['estimations'] = {}

        for i in range(len(self.targets)):
            if self.targets[i].file_name and self.targets[i].file_name not in self.audio_files:
                path = self.targets[i].file_name
            else:
                path = 'target%s' % i

            self.test_data['targets'].append(path)
            self.audio_files[path] = self.targets[i]

        for i in range(len(self.estimations)):
            source_estimations = []
            for j in range(len(self.estimations[i])):
                path = 'source%s_estimation%s' % (i, j)
                if (self.estimations[i][j].file_name
                        and self.estimations[i][j].file_name
                        not in self.audio_files):
                    path = self.estimations[i][j].file_name
                else:
                    path = 'source%s_estimation%s' % (i, j)

                source_estimations.append(path)
                self.audio_files[path] = self.estimations[i][j]

            self.test_data['estimations']['source_%s' % i] = source_estimations


    def create_config(self):
        """
        Creates a config JSON file
        """

        template = pkg_resources.read_text(beaqlejs, 'config_mushra.json')
        self.test_config = json.loads(template)
        self.test_config['MaxTestsPerRun'] = self.max_tests

        test_set = []
        for i in range(len(self.test_data['targets'])):
            target = {
                "Name": "Target %s" % i,
                "TestID": "target_%s" % i,
            }

            files = {"Reference": self.test_data['targets'][i]}

            j = 0
            for key in self.test_data['estimations']:
                files[str(j)] = self.test_data['estimations'][key][i]
                j += 1

            target['Files'] = files
            test_set.append(target)

        self.test_config['Testsets'] = test_set
        self.test_config = "var TestConfig = %s;" % json.dumps(self.test_config)
        self.test_config = bytes(self.test_config, 'utf-8')


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

        :meta private:
        """

        path = environ['PATH_INFO']

        if path in self.statics:
            response_body = self.statics[path]['data']()
            start_response('200 OK', Subjective.get_headers(response_body,
                                                            self.statics[path]['type']))

        elif path[1:] in self.audio_files:
            response_body = self.get_wav(path[1:])
            start_response('200 OK', Subjective.get_headers(response_body, 'audio/wav'))

        else:
            response_body = b''
            start_response('404 Not Found', Subjective.get_headers(response_body, 'text/html'))

        return [response_body]


    def post(self, environ, start_response):
        """
        Process a POST request

        :meta private:
        """

        # the environment variable CONTENT_LENGTH may be empty or missing
        path = environ['PATH_INFO']

        if path == '/finished_test':
            response_body = self.handle_finished_test(environ, start_response)

        else:
            response_body = b''
            start_response('404 Not Found', Subjective.get_headers(response_body, 'text/html'))

        return [response_body]


    @staticmethod
    def get_headers(response_body, mime_type):
        """
        Return headers for HTML

        :meta private:
        """

        response_headers = [
            ('Content-Type', mime_type),
            ('Content-Length', str(len(response_body)))
        ]

        return response_headers



    def get_wav(self, key):
        """
        Return a WAV file from dictionary of audio files for test

        :meta private:
        """

        audio = self.audio_files[key]
        wav = None
        with tempfile.TemporaryFile() as file_handle:
            scipy.io.wavfile.write(file_handle, audio.get_sample_rate(), audio.get_audio())
            wav = file_handle.read()

        return wav


    def handle_finished_test(self, environ, start_response):
        """
        Handle a finished test
        """

        try:
            request_body_size = int(environ.get('CONTENT_LENGTH', 0))
        except ValueError:
            request_body_size = 0

        # When the method is POST the variable will be sent
        # in the HTTP request body which is passed by the WSGI server
        # in the file like wsgi.input environment variable.
        request_body = environ['wsgi.input'].read(request_body_size)
        data = parse.parse_qs(request_body.decode('utf-8'))

        response = {
            'error': False,
            'message': "Data saved!"
        }

        test_results = data.get('testresults')
        test_results = json.loads(test_results[0])
        self.extract_scores(test_results)
        self._compute_stats()

        # Save test results as JSON files
        self.save_test_results(self.stats, 'subjective_stats.json')
        self.save_test_results(test_results, 'subjective_results.json')

        response_body = bytes(json.dumps(response), 'utf-8')
        start_response('200 OK', Subjective.get_headers(response_body, 'application/json'))
        return response_body


    def extract_scores(self, test_results):
        """
        Pulls out scores from test results and saves them into the scores attribute
        """

        self.scores = {}
        for score in test_results:
            test_id = score.get('TestID', None)
            if not test_id:
                continue

            rating = score.get('rating', None)
            if not rating:
                continue

            target_scores = {}
            for key in rating:
                if key == 'Reference':
                    target_scores['reference'] = {'results': rating[key]}

                else:
                    target_scores['source_%s' % key] = {'results': rating[key]}

            self.scores[test_id] = target_scores


    def save_test_results(self, test_results, file_name):
        """
        Save test results as JSON
        """

        file_name = os.path.join(self.output_dir, file_name)
        with open(file_name, 'w') as file_write:
            json.dump(test_results, file_write, indent=True)


    def setup_statics(self):
        """
        Creates a dictionary of static files and lambda functions that return
        binary data when that static file is required

        :meta private:
        """

        self.statics = {
            '/': {
                'type': 'text/html',
                'data': lambda: pkg_resources.read_binary(beaqlejs, 'index.html')
            },
            '/css/smoothness/jquery-ui-1.8.18.custom.css': {
                'type': 'text/css',
                'data': lambda: pkg_resources.read_binary(beaqlejs, 'jquery-ui-1.8.18.custom.css')
            },
            '/css/styles.css': {
                'type': 'text/css',
                'data': lambda: pkg_resources.read_binary(beaqlejs, 'styles.css')
            },
            '/js/beaqle.js': {
                'type': 'text/javascript',
                'data': lambda: pkg_resources.read_binary(beaqlejs, 'beaqle.js')
            },
            '/js/jquery.js': {
                'type': 'text/javascript',
                'data': lambda: pkg_resources.read_binary(beaqlejs, 'jquery.js')
            },
            '/js/jquery-ui.custom.min.js': {
                'type': 'text/javascript',
                'data': lambda: pkg_resources.read_binary(beaqlejs, 'jquery-ui.custom.min.js')
            },
            '/config/config_mushra.js': {
                'type': 'text/javascript',
                'data': lambda: self.test_config
            },
            '/img/ajax-loader.gif': {
                'type': 'image/gif',
                'data': lambda: pkg_resources.read_binary(beaqlejs, 'ajax-loader.gif')
            },
            '/img/scale_abs.png': {
                'type': 'image/png',
                'data': lambda: pkg_resources.read_binary(beaqlejs.img, 'scale_abs.png')
            },
            '/img/scale_abs_background.png': {
                'type': 'image/png',
                'data': lambda: pkg_resources.read_binary(beaqlejs.img,
                                                          'scale_abs_background.png')
            },
            '/css/smoothness/images/ui-bg_flat_75_ffffff_40x100.png': {
                'type': 'image/png',
                'data': lambda: pkg_resources.read_binary(beaqlejs.img,
                                                          'ui-bg_flat_75_ffffff_40x100.png')
            },
            '/css/smoothness/images/ui-bg_flat_65_ffffff_1x400.png': {
                'type': 'image/png',
                'data': lambda: pkg_resources.read_binary(beaqlejs.img,
                                                          'ui-bg_flat_65_ffffff_1x400.png')
            },
            '/css/smoothness/images/ui-bg_glass_75_e6e6e6_1x400.png': {
                'type': 'image/png',
                'data': lambda: pkg_resources.read_binary(beaqlejs.img,
                                                          'ui-bg_glass_75_e6e6e6_1x400.png')
            },
            '/css/smoothness/images/ui-bg_glass_75_dadada_1x400.png': {
                'type': 'image/png',
                'data': lambda: pkg_resources.read_binary(beaqlejs.img,
                                                          'ui-bg_glass_75_dadada_1x400.png')
            },
            '/css/smoothness/images/ui-bg_glass_65_ffffff_1x400.png': {
                'type': 'image/png',
                'data': lambda: pkg_resources.read_binary(beaqlejs.img,
                                                          'ui-bg_glass_65_ffffff_1x400.png')
            },
            '/css/smoothness/images/ui-bg_highlight-soft_75_cccccc_1x100.png': {
                'type': 'image/png',
                'data': lambda: pkg_resources.read_binary(
                    beaqlejs.img,
                    'ui-bg_highlight-soft_75_cccccc_1x100.png'
                )
            },

        }
