import pygame
import pyaudio
import sys
sys.path.append('/Users/odettweiler/OneDrive - Eastside Preparatory School/Documents/00 - Independent Study/Synthesizer/python files')
from synth_lib import *

# start init of synth

# initialize pyaudio stream for outputting audio
stream = pyaudio.PyAudio().open(
    rate=44100,
    channels=1,
    format=pyaudio.paInt16,
    output=True,
    frames_per_buffer=256
)

# init synth object, created with default values
synth = Synth()
# synth.debug = True # debug value outputs messages based on input

# pygame setup
pygame.init() # initialize display
synthDisplay = SynthDisplay(synth) # display object with methods

while synthDisplay.running: # indefinite loop
    synthDisplay.main_loop() # main display loop, showing all controls and waveform, etc.

    # synth code here
    if synth.notes: # sound is playing
        synth.output_frames(stream) # ouptut audio to stream
    if synth.midi_input.poll(): # notes are being inputted
        for event in synth.midi_input.read(num_events=16): # take each one
            synth.midi_in(event) # input into synth to be handled correctly

    # flip() the display to put your work on screen
    pygame.display.flip()

    synthDisplay.tick() # unlimited fps to allow synth to run well

pygame.quit() # quit application