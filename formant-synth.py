'''
formant-synth.py

Provides functions and formula calculations for producing formant speech synthesis.
Algorithm inspired by the 1980 Klatt Synthesizer, using similar input and variable values.

'''

'''
CLASSES
TODO: Param - Object containing parameters for synthesizer
TODO: Synth - synthesizer object, top-level
TODO: Section - section-level object, used for groups of parts of the synthesizer - cascade filters, voicing, etc - multiple Components in a row, essentially.
TODO: Component - Individual object for use in synthesizer, filter, amp, etc.
TODO: Voice -
TODO: Noise - 
TODO: Cascade -
TODO: Parallel -
TODO: Radiation -
TODO: Output -
TODO: Buffer -
TODO: Resonator -
TODO: Impulse -
TODO: Mixer -
TODO: Amplifier -
TODO: LowPass -
TODO: Normalizer -
TODO: NoiseGen -
TODO: Switch -
'''

import math
import numpy
import simpleaudio
import sys
from scipy.signal import resample_poly
from scipy.io.wavfile import write

def make(p = None):
    '''
    Creates a Synth object.
    The user can provide a Param object, but it is not required. A user can put in a parameter when they want to modify variables from the beginning.
    However, if no Param object is provided, a default Param object is used.
    @param p (Param) Param object used instead of default parameters. Defaults to None.
    @returns s (Synth) Synth object, ready to run(), attributes based on input.
    '''
    #use defaults if parameter not passed in
    if p is None:
        p = Param()
    #initialize Synth
    s = Synth()

    #loop through time-varying parameters
    for param in list(filter(lambda name: not name.startsWith("_"), dir(p))):
        if param is "BW" or param is "FF": #frequencies and bandwidths, in List form instead of just a number, need to be processed as Lists
            s.parameters[param]= [getattr(p,param)[i] for i in range(p.N_FORM)]
        else:
            s.parameters[param] = getattr(p,param)
    
    s.setup()
    return(s)

class Param(object):
    '''
    Parameters for synthesizer.
    @param F0 (float): Fundamental (Formant) frequency in Hz
    @param FF (list): List of floats, each one corresponds to a formant frequency in Hz
    @param BW (list): List of floats, each one corresponds to the bandwidth of a formant in Hz in terms of plus minus 3dB
    @param AV (float): Amplitude of voicing in dB
    @param AVS (float): Amplitude of quasi-sinusoidal voicing in dB
    @param AH (float): Amplitude of aspiration in dB
    @param AF (float): Amplitude of frication in dB
    @param SW (0 or 1): Controls switch from voicing waveform generator to cascade or parallel resonators
    @param FGP (float): Frequency of the glottal resonator 1 in Hz
    @param BGP (float): Bandwidth of glottal resonator 1 in Hz
    @param FGZ (float): Frequency of glottal zero in Hz
    @param BGZ (float): Bandwidth of glottal zero in Hz
    @param FNP (float): Frequency of nasal pole in Hz
    @param BNP (float): Bandwidth of nasal pole in Hz
    @param FNZ (float): Frequency on the nasal zero in Hz
    @param BNZ (float): Bandwidth of nasal zero in Hz
    @param BGS (float): Glottal resonator 2 bandwidth in Hz
    @param A1 (float): Amplitude of parallel formant 1 in Hz
    @param A2 (float): Amplitude of parallel formant 2 in Hz
    @param A3 (float): Amplitude of parallel formant 3 in Hz
    @param A4 (float): Amplitude of parallel formant 4 in Hz
    @param A5 (float): Amplitude of parallel formant 5 in Hz
    @param A6 (float): Amplitude of parallel formant 6 in Hz
    @param AN (float): Amplitude of nasal formant in dB
    NOTE: There's a lot of audio-technical jargon here. If you're confused, you're not alone, I just learned this too. 
    NOTE: https://docs.google.com/document/d/1lbwXXLnMClRgqMo7APaIjabHZJWW695wTqANjT9oeBE/edit?usp=sharing
    Each of the above parameters is stored in a Numpy array.
    '''

