#!/use/bin/env python
"""
This class runs a UDP server that receives OSC messages containing a path
to an audio file to use as a sound target for synthesizer sound matching. It returns
the parameter settings as JSON inside an OSC message.
"""

import os
import json
import socketserver
from copy import copy

import spiegelib as spgl
from spiegelib.network.osc import OscMessage, OscMessageBuilder


class UDPSocketHandler(socketserver.DatagramRequestHandler):
    """
    The RequestHandler class for the OSC server.
    """

    def handle(self):

        data = self.request[0].strip()
        osc_data = OscMessage(data)

        print("Received request:")
        print(osc_data.address, osc_data.params)

        response_builder = OscMessageBuilder("/error")
        response_builder.add_arg("unknown address")
        response = response_builder.build()

        if osc_data.address == "/sound_match":
            self.sound_matcher = self.server.sound_matcher
            response = self.try_sound_match(osc_data.params)

        socket = self.request[1]
        return_address = (self.client_address[0], self.server.send_port)
        socket.sendto(response.dgram, return_address)


    def try_sound_match(self, params):
        """
        Trys to run a sound match given a list of OSC parameters

        Args:
            params (list): List of OSC params
        """

        response = OscMessageBuilder()

        try:
            target = params[0]
        except IndexError:
            target = None

        if target is None or not isinstance(target, str):
            response.address = "/sound_match_error"
            response.add_arg("Invalid target string received")
            return response.build()

        # Load audio file to use as sound target
        target = os.path.abspath(target)
        audio = spgl.AudioBuffer()

        try:
            audio.load(target)
        except (RuntimeError, FileNotFoundError):
            response.address = "/sound_match_error"
            response.add_arg("Unable to load target audio file")
            return response.build()

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
            response.address = "/sound_match_error"
            response.add_arg(str(error))
            return response.build()

        # Successfully got a sound matched patch setting. Return as JSON string
        params = {'patch': params}
        params = json.dumps(params)
        response.address = "/sound_match_success"
        response.add_arg(params)
        return response.build()



class SoundMatchOSCServer():
    """
    Args:
        sound_matcher (:class:`~spiegelib.core.SoundMatch`): SoundMatch object to use
        host (str, optional): address to run server at. Defaults to 127.0.0.1
        receive (int, optonal): port to receive OSC messages on. Defaults to 9001.
        send (int, optional): port to send OSC messages on. Defaults to 9002.
    """

    def __init__(self, sound_matcher, host="127.0.0.1", receive=9001, send=9002):
        """
        Constructor
        """

        self.sound_matcher = sound_matcher
        self.host = host
        self.receive_port = receive
        self.send_port = send


    def start(self):
        """
        Start OSC server
        """

        server = socketserver.UDPServer((self.host, self.receive_port), UDPSocketHandler)
        server.send_port = self.send_port
        server.sound_matcher = self.sound_matcher
        print("Starting OSC server. Receiving on %s:%s. Sending on %s:%s"
              % (self.host, self.receive_port, self.host, self.send_port))
        print("To stop server press ctrl+c")

        server.serve_forever()
