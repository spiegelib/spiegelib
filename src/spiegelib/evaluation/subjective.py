#!/use/bin/env python
"""
A class for creating quick web-based listening tests for running trial subjective
evaluations. Can be used to create a localhost web application
that will serve a browser-based MUSHRA listening test using the
`BeaqleJS <https://github.com/HSU-ANT/beaqlejs>`_ framework.

Example:

    In this example we use folders of audio files to create a subjective listening test.
    Audio files have been generated from a synthesizer sound matching experiment
    with a genetic algorithm (GA) and a multi-layer perceptron (MLP).

    The folder ``folder_of_target_audio_files`` contains the set of target files,
    and the folders ``folder_of_ga_predictions`` and ``folder_of_mlp_predictions`` contain
    the audio results from sound matching for the GA and MLP.

    The target audio files must be labelled so they will be ordered correctly
    with natural sorting. For example: ``target_0.wav``, ``target_1.wav``, ..., etc.
    The prediction audio files must be labelled in a similar ordering so that
    they will be ordered to match the corresponding target file.

    .. code-block::

        import spiegelib as spgl

        targets = spgl.AudioBufer('./folder_of_target_audio_files')
        ga_predictions = spgl.AudioBuffer('./folder_of_ga_predictions')
        mlp_predictions = spgl.AudioBuffer('./folder_of_mlp_predictions')
        estimations = [ga_predictions, mlp_predictions]

        # Instantiate evaluation with audio buffers and run evaluation
        evaluation = spgl.evaluation.Subjective(targets, estimations)
        evaluation.evaluate()

    When the evaluate method is run, a localhost server will be started that will
    serve the listening test at ``localhost:8000``. When the test
    is submitted, the results will be saved as JSON files in the output directory
    set during object construction. These files will be named
    ``subjective_results.json`` and ``subjective_stats.json``.
    The save location defaults to the current working directory.
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
class Subjective(EvaluationBase):
    """
    Args:
        targets (list): a list of :ref:`AudioBuffers <audio_buffer>` to use as references
            in a MUSHRA style listening test
        estimations (list): list of lists of :ref:`AudioBuffers <audio_buffer>` to use
            as stimuli in a MUSHA style listening test. Each list of AudioBuffer objects
            should represent an audio source (such as audio from a particular estimator
            in a sound match experiment), and each AudioBuffer in the inner lists should
            be ordered to match the target sound to use as a referenece.
        output_dir (str, optional): directory to save test results. Defualts to current
            working directory.
        max_tests (int, optional): If set, will limit the number of test pages that are
            presented per subjective test. Defaults to showing a page for each target.
        show_results (bool, optional): Will show the test results on the end page of
            the test if set to True. Defaults to True.
        show_ids (bool, optional): Display Reference and source numbers for each track
            on each page. Use this if you want to know what the source of each track is,
            not meant for a true listening test.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, targets, estimations, output_dir='./',
                 max_tests=-1, show_results=True, show_ids=False):
        """
        Constructor
        """

        super().__init__(targets, estimations)

        self.test_data = {}
        self.port = 8000
        self.address = "localhost"
        self.max_tests = max_tests
        self.show_results = show_results
        self.show_ids = show_ids
        self.output_dir = os.getcwd() if output_dir == './' else output_dir

        self.setup_statics()
        self.setup_test()
        self.create_config()


    def evaluate(self):
        """
        Begin subjective test server. Starts up a web app that hosts the subjective
        evaluation test and serves it to localhost:8000 by default. Once the test has
        been completed and submitted from the browser, test results will be saved as
        JSON files (subjective_results.json & subjective_stats.json) to the output directory
        defined during construction (defaults to the current working directory).
        """

        with make_server(self.address, self.port, self) as httpd:
            print("Serving listening test at %s:%s" % (self.address, self.port))
            httpd.serve_forever()


    def evaluate_target(self, target, predictions):
        """
        Not used in the subjective evaluation
        """

    def verify_input_list(self, input_list):
        """
        Checks input targets and estimation objects
        """
        EvaluationBase.verify_audio_input_list(input_list)


    def setup_test(self):
        """
        Setup the test dictionary and also save all audio files into an audio
        file dictionary keyed on their path that will be requested.
        """

        self.audio_files = {}
        self.audio_folder = 'audio'
        self.test_data['targets'] = []
        self.test_data['estimations'] = {}

        for i, target in enumerate(self.targets):
            path = target.file_name
            if not path or path in self.audio_files:
                path = 'target_%s.wav' % i

            self.test_data['targets'].append(path)
            self.audio_files[path] = target

        for i, source in enumerate(self.estimations):
            source_estimations = []
            for j, estimation in enumerate(source):
                path = estimation.file_name
                if not path or path in self.audio_files:
                    path = 'source_%s_estimation_%s.wav' % (i, j)

                source_estimations.append(path)
                self.audio_files[path] = estimation

            self.test_data['estimations']['source_%s' % i] = source_estimations


    def create_config(self):
        """
        Creates a config JSON file
        """

        template = pkg_resources.read_text(beaqlejs, 'config_mushra.json')
        self.test_config = json.loads(template)
        self.test_config['MaxTestsPerRun'] = self.max_tests
        self.test_config['ShowResults'] = self.show_results
        self.test_config['ShowFileIDs'] = self.show_ids

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
        """

        response_headers = [
            ('Content-Type', mime_type),
            ('Content-Length', str(len(response_body)))
        ]

        return response_headers



    def get_wav(self, key):
        """
        Return a WAV file from dictionary of audio files for test
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
                'data': lambda: bytes(self.test_config, 'utf-8')
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
