from pygame import midi
import math
import itertools
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import pygame

# starting development establishing library methods and classes


class SynthHelp:
    @staticmethod
    def convert_midi_to_name(midi):
        return midi.midi_to_ansi_note(midi)

    @staticmethod
    def initialize_midi():
        try:
            midi.init()
            default_id = midi.get_default_input_id()
            midi_input = midi.Input(device_id=default_id)
            print(
                "Initialized midi input with midi device: ",
                midi.get_device_info(default_id),
            )
            return midi_input, default_id
        except midi.MidiException as e:
            print("No MIDI device found")
            return -1, -1

    @staticmethod
    def debug_midi_in(event):
        (status, thing1, thing2, _), _ = event
        if status == SynthHelp.null_status:
            return 0
        if status == SynthHelp.note_status:
            print("Note: ", thing1, " with velocity ", thing2)
            return 1
        elif status == SynthHelp.cc_status:
            print("CC: ", thing1, " with value: ", thing2)
            return 2
        elif status == SynthHelp.misc_status:
            print("Misc button: ", thing1, " with value: ", thing2)
            return 3
        else:
            print(
                "Unidentified input: status = ",
                status,
                ", value 1 = ",
                thing1,
                ", value 2 = ",
                thing2,
            )
            return 4

    def running_mean(x, windowSize):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

    # static values needed for standardization
    default_adsr = (0.1, 0.1, 0.5, 1)
    default_filter = (5000, 400)
    note_status = 152
    cc_status = 184
    misc_status = 191
    null_status = 248


class Oscillators:
    @staticmethod
    def get_sin_oscillator(freq=220, sample_rate=44100):
        increment = (2 * math.pi * freq) / sample_rate
        sin_array = (math.sin(v) for v in itertools.count(start=0, step=increment))
        return sin_array

    @staticmethod
    def get_square_oscillator(freq=220, sample_rate=44100):
        increment = (2 * math.pi * freq) / sample_rate
        sin_array = (
            np.sign(math.sin(v)) for v in itertools.count(start=0, step=increment)
        )
        return sin_array

    # deprecated sawtooth
    # def get_sawtooth_oscillator(freq=55, sample_rate=44100):
    #     increment = (2 * math.pi * freq)/ sample_rate
    #     return ((2 * math.atan(math.tan(math.sin(v) * 0.5))) for v in itertools.count(start=0, step=increment))
    #     # tan_array = np.tan(sin_array * 0.5)
    #     # return (2 * np.arctan(tan_array))

    @staticmethod
    def get_sawtooth_oscillator(freq=220, sample_rate=44100):  # custom sawtooth wave
        period = 1 / freq  # period of the function
        # increase is rise/run divided by change in x
        increase = (2 / period) / sample_rate
        val = 0.0  # current value
        while True:  # run indefinitely
            val = (val + increase) % 2  # mod 2 to keep in bounds, add increase
            yield val - 1  # give value -1 to center on x-axis

    @staticmethod
    def get_triangle_oscillator(freq=220, sample_rate=44100):  # custom sawtooth wave
        period = 1 / (2 * freq)  # period of the function
        # increase is rise/run divided by change in x
        increase = (2 / period) / sample_rate
        val = 0.0  # current value
        while True:  # run indefinitely
            if val + increase > 1 or val + increase < -1:
                increase = -increase  # flip increment
            val = val + increase  # add increase
            yield val  # give value -1 to center on x-axis

    @staticmethod
    def test_oscillator(osc, title):  # show waveform for an oscillator
        y = []  # y values init
        x = []  # x values init
        for i in range(500):  # get 500 samples
            y.append(next(osc))  # get next value from oscillator
            x.append(i / 44100)  # get x value in seconds

        y = np.array(y)
        if title:
            plt.title(title)
        else:
            plt.title("Plotting oscillator waveform")
        plt.plot(x, y, color="red", marker="o")
        plt.show()


class Note:
    def __init__(self, midi_note, velocity, adsr, osc_name):
        self.note_name = midi_note
        self.freq = midi.midi_to_frequency(midi_note)
        self.vel = velocity
        self.timer = 0.0
        self.osc_name = osc_name
        if osc_name == "SINE":
            self.osc = Oscillators.get_sin_oscillator(self.freq)
        elif osc_name == "SQUARE":
            self.osc = Oscillators.get_square_oscillator(self.freq)
        elif osc_name == "SAWTOOTH":
            self.osc = Oscillators.get_sawtooth_oscillator(self.freq)
        elif osc_name == "TRIANGLE":
            self.osc = Oscillators.get_triangle_oscillator(self.freq)

        self.note_on = True
        self.envelope = ADSREnvelope(adsr[0], adsr[1], adsr[2], adsr[3], self.vel)
        self.amp = 0.0

    def update_amp(self, dt):
        self.amp = self.envelope.get_amp(dt, self.note_on)
        if self.osc_name == "SQUARE":
            self.amp *= 0.4
        elif self.osc_name == "SAWTOOTH":
            self.amp *= 0.4
        elif self.osc_name == "TRIANGLE":
            self.amp *= 0.9

    def update_osc(self, osc_name):
        self.osc_name = osc_name
        if osc_name == "SINE":
            self.osc = Oscillators.get_sin_oscillator(self.freq)
        elif osc_name == "SQUARE":
            self.osc = Oscillators.get_square_oscillator(self.freq)
        elif osc_name == "SAWTOOTH":
            self.osc = Oscillators.get_sawtooth_oscillator(self.freq)
        elif osc_name == "TRIANGLE":
            self.osc = Oscillators.get_triangle_oscillator(self.freq)

    def get_next_osc_value(self):
        if self.osc:
            return next(self.osc)

    def get_samples(self, num_notes, amp=1, num_samples=256):
        return [self.get_single_sample(num_notes, amp) for _ in range(num_samples)]

    def get_single_sample(self, num_notes, amp=1):
        return int(self.get_next_osc_value() * 32767 * self.amp * amp / num_notes)


class ADSREnvelope:
    def __init__(self, attack, decay, sustain_ratio, release, velocity):
        self.attack = attack  # length of time before the full volume is reached
        self.max_volume = velocity / 127  # max volume between zero and one
        self.attack_slope = (
            self.max_volume / self.attack
        )  # get the slope of the attack line

        self.decay = decay  # length of time before it goes down to sustain volume

        # amplitude for sustain is max times ratio
        self.sustain = self.max_volume * sustain_ratio
        self.decay_slope = (
            self.sustain - self.max_volume
        ) / self.decay  # rise/run with the negative

        self.release = release  # time it takes for the sound to fade out
        self.release_slope = -self.sustain / self.release

        self.note_released = False
        self.phase = "ATTACK"
        self.timer = 0.0

    def get_amp(self, dt, note_on):
        if not note_on and not self.note_released:  # note has been released
            self.note_released = True
            if self.phase != "SUSTAIN":  # has not reached sustain phase yet
                cur_amp = self.get_amp(0, True)  # current volume
                # change slope of release line to be accurate
                self.release_slope = -cur_amp / self.release
            self.phase = "RELEASE"  # change phase
            self.timer = 0.0  # reset timer

        if self.phase == "ATTACK":
            self.timer += dt  # update timer

            if self.timer >= self.attack:  # attack is finished
                self.phase = "DECAY"  # change phase
                self.timer = 0.0  # reset timer
                return self.max_volume

            return self.timer * self.attack_slope  # use y = mx + b

        elif self.phase == "DECAY":
            self.timer += dt  # update timer

            if self.timer >= self.decay:  # decay is done
                self.phase = "SUSTAIN"  # change phase
                self.timer = 0.0  # reset timer
                return self.sustain

            # use y = mx + b
            return (self.decay_slope * self.timer) + self.max_volume

        elif self.phase == "SUSTAIN":
            return self.sustain  # sustain always returns the same volume

        elif self.phase == "RELEASE":
            self.timer += dt  # update timer

            if self.timer >= self.release:
                return -1  # return invalid is note is gone

            # use y = mx + b
            return (self.release_slope * self.timer) + self.sustain

    def update_attack(self, attack_):  # takes input between 1-127
        attack_val = (attack_ / 127) * 3  # normalize between 0-1
        if attack_val < 0.01:
            attack_val = 0.01  # adding bounds
        if attack_val > 3:
            attack_val = 3
        print("Envelope attack (s): ", self.attack)
        self.attack = attack_val  # length of time before the full volume is reached
        self.attack_slope = (
            self.max_volume / self.attack
        )  # get the slope of the attack line

    def update_decay(self, decay_):  # takes input between 1-127
        decay_val = (decay_ / 127) * 3  # normalize between 0-1 then up till three
        if decay_val < 0.01:
            decay_val = 0.01  # adding bounds
        if decay_val > 3:
            decay_val = 3
        print("Envelope decay (s): ", self.decay)
        self.decay = decay_val  # length of time before it goes down to sustain volume
        self.decay_slope = (
            self.sustain - self.max_volume
        ) / self.decay  # rise/run with the negative

    def update_sustain(self, sustain_ratio):  # takes input between 1-127
        sustain_val = sustain_ratio / 127  # normalize between 0-1
        if sustain_val < 0.01:
            decay_val = 0.01  # adding bounds
        if sustain_val > 1:
            decay_val = 1
        # amplitude for sustain is max times ratio
        self.sustain = self.max_volume * sustain_val
        print("Envelope sustain (0-1): ", self.sustain)
        self.decay_slope = (
            self.sustain - self.max_volume
        ) / self.decay  # rise/run with the negative
        self.release_slope = -self.sustain / self.release

    def update_release(self, release_):
        release_val = (release_ / 127) * 5  # normalize between 0-1
        if release_val < 0.01:
            release_val = 0.01  # adding bounds
        self.release = release_val  # time it takes for the sound to fade out
        print("Envelope release (s): ", self.release)
        self.release_slope = -self.sustain / self.release

    def get_values(self):
        return self.attack, self.decay, self.sustain, self.release

    def get_attack_draw_val(self):
        return (self.attack / 2.9) + 0.01

    def get_attack_draw_text(self):
        return str(round(self.attack))

        +"s"

    def get_sustain_draw_val(self):
        return self.sustain / self.max_volume

    def get_sustain_draw_text(self):
        return str(round(self.sustain / self.max_volume))

    def get_decay_draw_val(self):
        return (self.decay / 2.9) + 0.01

    def get_decay_draw_text(self):
        return round(self.decay) + "s"

    def get_release_draw_val(self):
        return (self.release / 4.9) + 0.01

    def get_release_draw_text(self):
        return round(self.release) + "s"

    def get_coords(self):
        points = (
            []
        )  # x and y points for vis, x is in seconds, y is normalized between 0-1
        points.append(pygame.math.Vector2(0, 0))  # first
        # attack for time (x) and max height
        points.append(pygame.math.Vector2(self.attack, 1))
        # offset by last x, y is normalized sustain
        points.append(
            pygame.math.Vector2(
                points[1].x + self.decay, self.sustain / self.max_volume
            )
        )
        # offset by last x, y is normalized sustain
        points.append(pygame.math.Vector2(points[2].x, self.sustain / self.max_volume))
        val_sum = self.attack + self.decay + self.release  # total amount for all vars
        for i in range(15):  # total things
            if val_sum > i:  # greater than the current amount
                # add a certain amount to the x of the sustain bar
                points[3].x += 0.5
        # final value, release value for x and 0 for y
        points.append(pygame.math.Vector2(points[3].x + self.release, 0))

        return points


class EQFilter:
    def __init__(self, low_freq, high_freq, filter_on=True):
        self.low_freq = low_freq
        # self.highpass_slope = high_pass_slope if high_pass_slope < 0 else -high_pass_slope # this should point downwards, negative slope
        self.high_freq = high_freq
        # self.lowpass_slope = low_pass_slope if low_pass_slope > 0 else -low_pass_slope # this should point upwards, positive slope
        self.on = filter_on

        self.MAX_FREQ = 10000  # highest frequency possible, lowest is 0

    def update_high_freq(self, value):
        freq = (value / 127) * self.MAX_FREQ  # map between zero and max frequency
        self.high_freq = freq if freq > self.low_freq else self.low_freq + 1
        print("Filter high frequency (Hz): ", self.high_freq)

    def get_high_freq_draw_val(self):
        return self.high_freq / self.MAX_FREQ

    def get_high_freq_draw_text(self):
        return round(self.high_freq) + "Hz"

    # def update_high_slope(self, value):
    #     slope = ((value/127)*500)+0.1
    #     self.highpass_slope = -slope

    def update_low_freq(self, value):
        freq = (value / 127) * self.MAX_FREQ  # map between zero and max frequency
        self.low_freq = freq if freq < self.high_freq else self.high_freq - 1
        print("Filter low frequency (Hz): ", self.low_freq)

    def get_low_freq_draw_val(self):
        return self.low_freq / self.MAX_FREQ

    def get_low_freq_draw_text(self):
        return round(self.low_freq) + "Hz"

    # def update_high_slope(self, value):
    #     slope = ((value/127)*500)+0.1
    #     self.lowpass_slope = slope

    # def get_freq_amplitude(self, freq):
    #     if freq < self.lowpass_freq:
    #         return 0
    #     elif freq >= self.lowpass_freq and freq < self.highpass_freq:
    #         return 1
    #     elif freq >= self.highpass_freq:
    #         return 0

    # def low_filter_chunk(self, CHUNK):
    #     w = np.linspace(0,CHUNK-2,CHUNK)
    #     return 1*(w <= self.high_freq)

    # def high_filter(self, CHUNK):
    #     w = np.linspace(0,CHUNK-2,CHUNK)
    #     return 1*(w >= self.low_freq)

    # def filter_chunk(self, CHUNK):
    #     w = np.linspace(0,CHUNK-2,CHUNK)
    #     return 1*((w >= self.low_freq) & (w<=self.high_freq))

    def low_pass_filter(self, chunk):
        freqRatio = self.high_freq / 44100
        N = int(math.sqrt(0.196201 + freqRatio**2) / freqRatio)
        filtered = SynthHelp.running_mean(chunk, N)
        return filtered


"""
important values for input mappings: (for launchkey mini)
status = 152 for note on and off, velocity = 0 for off
status = 153 for pads
status = 184 for knobs, cc numbers are 21-28, they go through 1-127
"""


class ControlMap:
    def __init__(
        self, cc_, change_variable_, get_variable_, get_text_, page_, name_="unnamed"
    ):
        self.cc = cc_
        self.page = page_
        self.name = name_
        self.change_variable = change_variable_
        self.get_variable = get_variable_
        self.get_text = get_text_

    def change_update_func(self, func):
        self.change_variable = func

    def change_get_func(self, func):
        self.get_variable = func

    def change_get_text(self, func):
        self.get_text = func


# synth class here


class Synth:
    # init with default envelope, filter, and waveform
    def __init__(
        self,
        default_adsr=(0.1, 0.1, 0.5, 1),
        default_filter_vals=(1000, 10, 400, 10),
        waveform_="SINE",
        filter_on=False,
    ):
        self.notes = {}  # note dictionary for notes being played or fading
        self.debug = False
        self.last_time = time.perf_counter()  # current time for timing
        self.default_adsr = ADSREnvelope(
            default_adsr[0], default_adsr[1], default_adsr[2], default_adsr[3], 127
        )  # default envelope
        self.waveform = waveform_
        self.total_amp = 1
        self.filter = EQFilter(
            default_filter_vals[0], default_filter_vals[1], filter_on
        )

        self.midi_input, self.default_id = SynthHelp.initialize_midi()

        self.control_page = 1
        self.control_map = [  # page 1 of control map
            ControlMap(
                21,
                self.default_adsr.update_attack,
                self.default_adsr.get_attack_draw_val,
                self.default_adsr.get_attack_draw_text,
                1,
                "Attack",
            ),  # knob 1 for attack
            ControlMap(
                22,
                self.default_adsr.update_decay,
                self.default_adsr.get_decay_draw_val,
                self.default_adsr.get_decay_draw_text,
                1,
                "Decay",
            ),  # knob 2 for decay
            ControlMap(
                23,
                self.default_adsr.update_sustain,
                self.default_adsr.get_sustain_draw_val,
                self.default_adsr.get_sustain_draw_text,
                1,
                "Sustain",
            ),  # knob 3 for sustain
            ControlMap(
                24,
                self.default_adsr.update_release,
                self.default_adsr.get_release_draw_val,
                self.default_adsr.get_release_draw_text,
                1,
                "Release",
            ),  # knob 4 for release
            ControlMap(
                25,
                self.filter.update_low_freq,
                self.filter.get_low_freq_draw_val,
                self.filter.get_high_freq_draw_text,
                1,
                "HPF frequency",
            ),  # knob 4 for release
            ControlMap(
                26,
                self.filter.update_high_freq,
                self.filter.get_high_freq_draw_val,
                self.filter.get_high_freq_draw_text,
                1,
                "LPF frequency",
            ),  # knob 4 for release
            ControlMap(
                27,
                self.update_waveform,
                self.get_waveform_draw_val,
                self.get_waveform,
                1,
                "Waveform",
            ),  # knob 7 for waveform
            ControlMap(
                28,
                self.update_total_amp,
                self.get_total_amp_draw_val,
                self.get_total_amp_draw_text,
                1,
                "Total volume",
            ),
            # page 2
            ControlMap(
                27,
                self.update_buffer_length,
                self.get_buffer_len_draw_val,
                self.get_buffer_len_draw_text,
                2,
                "Buffer length",
            ),
            ControlMap(
                28,
                self.update_total_amp,
                self.get_total_amp_draw_val,
                self.get_total_amp_draw_text,
                2,
                "Total volume",
            ),
        ]  # knob 8 for total volume (both pages)

        self.buffer_len = 288  # buffer length
        self.buffer = [self.buffer_len]  # max buffer length

    # deprecated adsr update
    # def update_adsr(self, adsr): # update adsr
    #     self.default_adsr = adsr # update variable

    def update_total_amp(self, amp):  # update amp for all notes
        self.total_amp = amp / 127  # update
        print("Synth total amplitude (0-1): ", self.total_amp)

    def get_total_amp_draw_val(self):
        return self.total_amp

    def get_total_amp_draw_text(self):
        return str(round(self.total_amp))

    def update_buffer_length(self, value):  # update buffer length
        # normalized to 0-1, multiply by frame length, up to 4 times that, plus 1 so at least one frame is displayed
        self.buffer_len = int(((value / 127) * 256) + 32)
        print("Buffer length: ", self.buffer_len)

    def get_buffer_len_draw_val(self):
        return self.buffer_len / 288

    def get_buffer_len_draw_text(self):
        return str(round(self.buffer_len))

    # update waveform for all notes, takes in value between 1 and 127
    def update_waveform(self, value, updateNotes=True):
        # increment for determining waveform, based on 127 as max value (divide 127 into 4 quadrants)
        inc = int(127 / 4)
        changed = False
        if value <= inc and self.waveform != "SINE":  # sine wave
            self.waveform = "SINE"
            changed = True
        elif (
            value > inc and value <= inc * 2 and self.waveform != "SQUARE"
        ):  # square wave
            self.waveform = "SQUARE"
            changed = True
        elif (
            value > inc * 2 and value <= inc * 3 and self.waveform != "SAWTOOTH"
        ):  # saw wave
            self.waveform = "SAWTOOTH"
            changed = True
        elif (
            value > inc * 3 and value <= 127 and self.waveform != "TRIANGLE"
        ):  # tri wave
            self.waveform = "TRIANGLE"
            changed = True

        if changed:
            print("Oscillator waveform: ", self.waveform)
            if updateNotes:
                for note in self.notes.items():
                    note[1].update_osc(self.waveform)

    def get_waveform_draw_val(self):
        if self.waveform == "SINE":
            return 0
        if self.waveform == "SQUARE":
            return 0.33
        if self.waveform == "SAWTOOTH":
            return 0.66
        if self.waveform == "TRIANGLE":
            return 1

    def get_waveform(self):
        return self.waveform

    def delta_time(self):  # get difference in time since this was last called
        current_time = time.perf_counter()
        dt = current_time - self.last_time
        self.last_time = current_time  # return value
        return dt  # return delta

    # update every note amplitude based on envelopes
    def update_all_note_amplitudes(self):
        dt = self.delta_time()
        for note in self.notes.items():  # each note
            note[1].update_amp(dt)  # call method

    def get_all_samples(self, num_samples=256):  # get all of the samples for
        if (
            self.notes
        ):  # there are notes being played, sum all notes being played for the number of samples needed
            sum_chunk = [
                sum(
                    [
                        note.get_single_sample(len(self.notes.items()), self.total_amp)
                        for note in self.notes.values()
                    ]
                )
                for _ in range(num_samples)
            ]
            if self.filter.on:
                return self.filter.low_pass_filter(sum_chunk)
            else:
                return sum_chunk
        else:  # no sound being played, return 0 for every sample
            return [0 for _ in range(num_samples)]

    def update_buffer(self, frame):
        self.buffer.append(frame)
        while len(self.buffer) > self.buffer_len:  # greater than max size
            del self.buffer[0]  # delete last frame

    def note_in(self, note_event):  # handle midi input and update note dictionary
        (status, note, vel, _), _ = note_event
        # print(note_event) # print out event
        if note in self.notes:  # if note's already being played
            if vel == 0:  # turn note off
                self.notes[note].note_on = False  # update value
                return
            else:  # play note again
                self.notes[note] = Note(
                    note, vel, self.default_adsr.get_values(), self.waveform
                )  # reinitialize
                return
        if note not in self.notes:  # if note is not in the dict, put it in
            self.notes[note] = Note(
                note, vel, self.default_adsr.get_values(), self.waveform
            )  # init with note value, velocity, and envelope
            return
        # print(self.notes)

    # deprecated, use mapped_control_in instead
    def control_in(self, cc_event):  # handle midi input and update note dictionary
        (status, cc, value, _), _ = cc_event  # replace this??
        # print(cc_event) # print out event
        if cc == 21:  # attack
            self.default_adsr.update_attack(value)
        elif cc == 22:  # decay
            self.default_adsr.update_decay(value)
        elif cc == 23:  # sustain
            self.default_adsr.update_sustain(value)
        elif cc == 24:  # release
            self.default_adsr.update_release(value)
        elif cc == 27:  # change waveform
            self.update_waveform(value)
        elif cc == 28:  # amplitude adjustment
            self.update_total_amp(value)  # map between 0 and 1
        else:
            print("unmapped control change control change")

    # map a new control or remap an old one
    def map_control(self, cc, page, update, get_var, get_text):
        for control_mapping in self.control_map:
            if (
                control_mapping.cc == cc and control_mapping.page == page
            ):  # mapping already exists for this value
                if update and get_var and get_text:  # new update function
                    control_mapping.change_update_func(update)
                    control_mapping.change_get_func(get_var)
                    control_mapping.change_get_text(get_text)
                else:  # None value, destroy mapping
                    self.control_map.remove(control_mapping)
                return

        # control has not yet been mapped
        self.control_map.append(ControlMap(cc, page, update))

    # handle input for misc buttons
    def misc_in(self, misc_event):
        (status, num, value, _), _ = misc_event
        if num == 104:  # control page 1 button
            if value == 127 and self.control_page == 2:  # needs to be updated
                self.control_page = 1
                print("Entering control page 1")
        elif num == 105:  # control page 2 button
            if value == 127 and self.control_page == 1:  # needs to be updated
                self.control_page = 2
                print("Entering control page 2")
        elif num == 117:  # toggle debug mode
            if value == 127:  # time to flip
                self.debug = not self.debug  # flip value
                if self.debug:
                    print("Entering debug mode...")
                else:
                    print("Exiting debug mode...")

        # print(self.control_page)

    # handle input for a mapped control
    def mapped_control_in(self, cc_event):
        (status, cc, value, _), _ = cc_event
        # print(cc_event) # print out event
        for mapping in self.control_map:
            if (
                mapping.cc == cc and mapping.page == self.control_page
            ):  # correct cc and current page is the same
                mapping.change_variable(value)
                return 1

        print("Control not yet mapped: ", cc, self.control_page, value)
        return -1

    # most important functions, called in the main loop
    # pull everything together with input and output respectively

    def output_frames(self, stream):
        self.update_all_note_amplitudes()

        to_remove = []
        for note in self.notes.items():  # check each note
            if note[1].amp < 0.0:  # amp == -1, note has faded out
                to_remove.append(note)  # prepare to remove
        for note in to_remove:  # remove all faded notes
            del self.notes[note[0]]

        samples = self.get_all_samples()  # get all sounds being played
        [self.update_buffer(sample) for sample in samples]  # add each sample

        samples = np.int16(samples).tobytes()  # convert to correct format
        stream.write(samples)  # output sound

    def midi_in(self, event):
        if self.debug:
            SynthHelp.debug_midi_in(event)
        if event[0][0] == SynthHelp.note_status:  # status is that note is played
            self.note_in(event)  # put into synth code
        elif event[0][0] == SynthHelp.cc_status:  # status is that control is changed
            self.mapped_control_in(event)  # put into synth code
        elif event[0][0] == SynthHelp.misc_status:  # status is misc button
            self.misc_in(event)


class SynthDisplay:
    def __init__(self, synth):
        self.height = 600
        self.width = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Python Synthesizer")
        self.clock = pygame.time.Clock()
        self.running = True
        self.synth = synth

        self.BG_COLOR = (92, 191, 148)
        self.MINI_COLOR = (102, 212, 164)
        self.TEXT_COLOR = (207, 207, 207)
        self.TEXT_SIZE = 16
        self.font = pygame.font.Font(None, self.TEXT_SIZE)
        self.CONTROL_COLOR_1 = (99, 99, 99)
        self.CONTROL_COLOR_2 = (58, 148, 121)
        self.DEBUG_GRAY_COLOR = (189, 189, 189)
        self.DEBUG_RED_COLOR = (255, 36, 36)

        self.MINI_HEIGHT = 200
        self.MINI_WIDTH = 300

    def tick(self):
        self.clock.tick()

    def show_waveform(self, buffer, xoff, yoff):
        # draw background
        pygame.draw.rect(
            self.screen,
            self.MINI_COLOR,
            [xoff, yoff, xoff + self.MINI_WIDTH, yoff + self.MINI_HEIGHT],
        )

        if len(buffer) < 2:
            return
        x_inc = self.MINI_WIDTH / len(buffer)
        cur_x = xoff
        points = [
            pygame.math.Vector2(
                xoff,
                ((buffer[i] / 32767) * self.MINI_HEIGHT / 2)
                + yoff
                + self.MINI_HEIGHT / 2,
            )
            for i in range(len(buffer))
        ]
        for pt in points:
            pt.x = cur_x
            cur_x += x_inc

        pygame.draw.aalines(self.screen, pygame.color.Color(0, 0, 0), False, points)

    def show_adsr(self, adsr, xoff, yoff):
        # draw background
        pygame.draw.rect(
            self.screen,
            self.MINI_COLOR,
            [xoff, yoff, self.MINI_WIDTH, self.MINI_HEIGHT],
        )

        points = adsr.get_coords()
        for pt in points:
            pt.x = ((pt.x / points[-1].x) * self.MINI_WIDTH) + xoff
            pt.y = self.MINI_HEIGHT + yoff - (pt.y * self.MINI_HEIGHT)

        pygame.draw.aalines(self.screen, pygame.color.Color(0, 0, 0), False, points)

    def show_control(self, mapping, xoff, yoff):
        text1 = self.font.render(mapping.name, True, self.TEXT_COLOR)
        textRect1 = text1.get_rect()
        textRect1.center = (xoff + 20, yoff - 20)
        self.screen.blit(text1, textRect1)

        # text2 = self.font.render(mapping.get_text(), True, self.TEXT_COLOR)
        # textRect2 = text2.get_rect()
        # textRect2.center = (xoff + 20, yoff + 20)
        # self.screen.blit(text2, textRect2)

        # draw first arc
        pygame.draw.arc(
            self.screen,
            self.CONTROL_COLOR_2,
            pygame.rect.Rect(xoff, yoff, 30, 30),
            0,
            math.pi,
            1,
        )

        # draw second arc, with mapped angle for value
        mapped_angle = (-math.pi * mapping.get_variable()) + math.pi
        pygame.draw.arc(
            self.screen,
            self.CONTROL_COLOR_1,
            pygame.rect.Rect(xoff, yoff, 30, 30),
            mapped_angle,
            math.pi,
            6,
        )

    def show_all_controls(self):
        current_x = 20  # start here
        for cc in range(21, 29):  # for each knob
            for mapping in self.synth.control_map:  # check each mapping
                if (
                    mapping.cc == cc and mapping.page == self.synth.control_page
                ):  # if we find the correct one,
                    # show it on screen
                    self.show_control(mapping, current_x, self.height - 100)
            # increment each time we move on from a knob to have correct spacing
            current_x += self.width / 8

        # display page number
        text = self.font.render(
            "Page " + str(self.synth.control_page), True, self.TEXT_COLOR
        )
        textRect = text.get_rect()
        textRect.center = (40, self.height - 20)
        self.screen.blit(text, textRect)

    def show_debug(self, xoff, yoff):
        # show appropriate color based on whether debug is on
        if self.synth.debug:  # red for debug
            pygame.draw.circle(self.screen, self.DEBUG_RED_COLOR, [xoff, yoff], 5)
        else:  # gray for normal
            pygame.draw.circle(self.screen, self.DEBUG_GRAY_COLOR, [xoff, yoff], 5)

    def show_total_amp(self, xoff):
        yoff = 15
        pygame.draw.line(
            self.screen, self.CONTROL_COLOR_1, [xoff, yoff], [xoff, self.MINI_HEIGHT], 1
        )
        amp_height = self.synth.get_total_amp_draw_val() * (self.MINI_HEIGHT - yoff)
        pygame.draw.rect(
            self.screen,
            self.CONTROL_COLOR_1,
            [xoff - 10, self.MINI_HEIGHT - amp_height, 20, 10],
        )

    def show_waveform_title(self, xoff, yoff):
        pygame.draw.rect(self.screen, self.CONTROL_COLOR_1, [xoff, yoff, 20, 20])
        text = self.font.render(
            "Waveform: " + str(self.synth.get_waveform()), True, self.TEXT_COLOR
        )
        textRect = text.get_rect()
        textRect.center = (xoff + 10, yoff + 10)
        self.screen.blit(text, textRect)

    def main_loop(self):
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:  # mouse is clicked
                mPos = pygame.mouse.get_pos()
                if (
                    mPos[0] >= self.width - 40 and mPos[1] <= self.height - 40
                ):  # within debug spot
                    self.synth.debug = not self.synth.debug  # flip debug

        # fill the screen with a color to wipe away anything from last frame
        self.screen.fill(self.BG_COLOR)

        # display waveform visualization
        self.show_waveform(self.synth.buffer, 0, 0)
        self.show_adsr(self.synth.default_adsr, 315, 0)  # display adsr lines
        self.show_all_controls()  # show each knob in the control map
        self.show_total_amp(self.width - 75)  # show synth amplitude
        # place in center-left of screen
        self.show_waveform_title(40, self.height / 2)
        self.show_debug(self.width - 30, self.height - 30)  # display debug thingy
