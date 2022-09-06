'''
formant-synth.py

Provides functions and formula calculations for producing formant speech synthesis.
Algorithm inspired by the 1980 Klatt Synthesizer, using similar input and variable values.


Author: Joey Ross
'''

'''
NOTE ABOUT DOCSTRINGS
----------------------------------

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
    @param F0   (float): Fundamental (Formant) frequency in Hz
    @param FF   (list): List of floats, formant frequencies (Hz)
    @param BW   (list): List of floats, bandwidth of a formant (Hz) (+/- 3dB)
    @param AV   (float): Amplitude of voicing (dB)
    @param AQSV (float): Amplitude of quasi-sinusoidal voicing (dB)
    @param AA   (float): Amplitude of aspiration (dB)
    @param AF   (float): Amplitude of frication (dB)
    @param CS   (int 0 or 1): Toggle between voicing waveform generator and cascade or parallel resonators
    @param FGR1 (float): Frequency of the glottal resonator 1 in Hz
    @param BGR1 (float): Bandwidth of glottal resonator 1 in Hz
    @param FGZ  (float): Frequency of glottal zero in Hz
    @param BGZ  (float): Bandwidth of glottal zero in Hz
    @param FNP  (float): Frequency of nasal pole in Hz
    @param BNP  (float): Bandwidth of nasal pole in Hz
    @param FNZ  (float): Frequency on the nasal zero in Hz
    @param BNZ  (float): Bandwidth of nasal zero in Hz
    @param BGR2 (float): Glottal resonator 2 bandwidth in Hz
    @param PF1  (float): Amplitude of parallel formant 1 in Hz
    @param PF2  (float): Amplitude of parallel formant 2 in Hz
    @param PF3  (float): Amplitude of parallel formant 3 in Hz
    @param PF4  (float): Amplitude of parallel formant 4 in Hz
    @param PF5  (float): Amplitude of parallel formant 5 in Hz
    @param PF6  (float): Amplitude of parallel formant 6 in Hz
    @param NF   (float): Amplitude of nasal formant in dB
    NOTE: There's a lot of audio-technical jargon here. If you're confused, you're not alone, I just learned this too. 
    NOTE: https://docs.google.com/document/d/1lbwXXLnMClRgqMo7APaIjabHZJWW695wTqANjT9oeBE/edit?usp=sharing
    Each of the above parameters is stored in a Numpy array.
    '''
    def __init__(self, FS=10000, N_FORM=5, DUR=1, F0=100,
                       FF=[500, 1500, 2500, 3500, 4500],
                       BW=[50, 100, 100, 200, 250],
                       AV=60, AQSV=0, AA=0, AF=0,
                       CS=0, FGR1=0, BGR1=100, FGZ=1500, BGZ=6000,
                       FNP=250, BNP=100, FNZ=250, BNZ=100, BGR2=200,
                       PF1=0, PF2=0, PF3=0, PF4=0, PF5=0, PF6=0, NF=0):
        self.FS = FS
        self.DUR = DUR
        self.N_FORM = N_FORM
        self.N_SAMP = round(FS*DUR)
        
        self.VER = "KLSYN80" #TODO: Delete this?
        '''
        HEY YOU
        YES YOU
        THIS IS THE PLACE WHERE YOU PROBABLY MESSED UP
        LOOK HERE

        LOOK
        HERE

        LOOK HERE

        L
        O
        O
        K

        H
        E
        R
        E
        '''
        self.DT = 1/FS
        self.F0 = numpy.ones(self.N_SAMP)*F0
        self.FF = [numpy.ones(self.N_SAMP)*FF[i] for i in range(N_FORM)]
        self.BW = [numpy.ones(self.N_SAMP)*BW[i] for i in range(N_FORM)]
        self.AV = numpy.ones(self.N_SAMP)*AV
        self.AQSV = numpy.ones(self.N_SAMP)*AQSV
        self.AA = numpy.ones(self.N_SAMP)*AA
        self.AF = numpy.ones(self.N_SAMP)*AF
        self.FNZ = numpy.ones(self.N_SAMP)*FNZ
        self.CS = numpy.ones(self.N_SAMP)*CS
        self.FGR1 = numpy.ones(self.N_SAMP)*FGR1
        self.BGR1 = numpy.ones(self.N_SAMP)*BGR1
        self.FGZ = numpy.ones(self.N_SAMP)*FGZ
        self.BGZ = numpy.ones(self.N_SAMP)*BGZ
        self.FNP = numpy.ones(self.N_SAMP)*FNP
        self.BNP = numpy.ones(self.N_SAMP)*BNP
        self.BNZ = numpy.ones(self.N_SAMP)*BNZ
        self.BGR2 = numpy.ones(self.N_SAMP)*BGR2
        self.PF1 = numpy.ones(self.N_SAMP)*PF1
        self.PF2 = numpy.ones(self.N_SAMP)*PF2
        self.PF3 = numpy.ones(self.N_SAMP)*PF3
        self.PF4 = numpy.ones(self.N_SAMP)*PF4
        self.PF5 = numpy.ones(self.N_SAMP)*PF5
        self.PF6 = numpy.ones(self.N_SAMP)*PF6
        self.NF = numpy.ones(self.N_SAMP)*NF

class Synth(object):
    """
    Synthesizes speech.
    @param name (string): Name of this synthesizer
    @param output (None): Output vector for the synthesizer, set by setup() later
    @param sections (None): List of sections in the synthesizer, set by setup() later
    @param parameters (dictionary): Dictionary of parameters
    Methods:
        setup(void): Run after parameter values are set, initializes synthesizer
        run(void): Clears current output vector and runs synthesizer
        play(void): Plays output via sounddevice module
        save(void): saves waveform to disk.
        _get_16_16K(int16): Converts output waveform to form for playing
    NOTE
    Synth contains all the synthesis parameters in an attribute called parameters. 
    The synthesis is organized around the concepts of sections and components.
    Sections are objects which represent components, organized in a certain way.
    Each section is composed of multiple components, which are small signal processing units like individual filters, resonators, amplifiers, etc. Each section has a run() /
        method which performs the operation that section is designed to do. For example, a Voice section's run() method generates a voicing waveform.
    One important caveat is that any time-varying parameters (i.e., all parameters except those labelled "synth settings" below) should be numpy arrays of length N_SAMP.
    Synth, while designed around the original Klatt formant synthesizer algorithm, is designed to be more flexible.
    /NOTE

    CITATION: Klatt, D. H. (1980). Software for a cascade/parallel formant synthesizer. The Journal of the Acoustical Society of America, 67 (3)
    """
    def __init__(self):

        # Create name
        self.name = "Formant Synthesizer"

        # Create empty attributes
        self.out = None
        self.sections = None

        # Create synthesis parameters dictionary
        param_list = ["F0", "AV", "OQ", "SQ", "TL", "FL",       # Source
                      "DI", "AQSV", "AV", "AF", "AA",           # Source
                      "FF", "BW",                               # Formants
                      "FGR1", "BGR1", "FGZ", "BGZ", "BGR2",        # Glottal pole/zero
                      "FNP", "BNP", "FNZ", "BNZ",               # Nasal pole/zero
                      "FTP", "BTP", "FTZ", "BTZ",               # Tracheal pole/zero
                      "A2F", "A3F", "A4F", "A5F", "A6F",        # Frication parallel
                      "B2F", "B3F", "B4F", "B5F", "B6F",        # Frication parallel
                      "A1V", "A2V", "A3V", "A4V", "ATV",        # Voicing parallel
                      "PF1", "PF2", "PF3", "PF4", "PF5", "NF",  # 1980 parallel
                      "ANV",                                    # Voicing parallel
                      "CS", "INV_SAMP",                         # Synth settings
                      "N_SAMP", "FS", "DT", "VER"]              # Synth settings
        self.parameters = {param: None for param in param_list}

    def setup(self):
        """
        Sets up Synth.
        Run after parameter values are set. 
        Initializes output vector & sections.
        """
        # Initialize data vectors
        self.out = numpy.zeros(self.parameters["N_SAMP"])

        #XXX: Versions are a bad idea but it's a little too difficult to remove this at this time. 
        #XXX: I'll probably leave it in for the submission of artifacts for this sprint.
        #TODO: Remove versions?
        if self.parameters["VER"] == "KLSYN80":
            # Initialize sections
            self.voice = Voice(self)
            self.noise = Noise(self)
            self.cascade = Cascade(self)
            self.parallel = Parallel(self)
            self.radiation = Radiation(self)
            self.output_module = OutputModule(self)
            # Create section-level connections
            self.voice.connect([self.cascade, self.parallel])
            self.noise.connect([self.cascade, self.parallel])
            self.cascade.connect([self.radiation])
            self.parallel.connect([self.radiation])
            self.radiation.connect([self.output_module])
            # Put all section objects into self.sections for reference
            self.sections = [self.voice, self.noise, self.cascade,
                             self.parallel, self.radiation, self.output_module]
            # Patch all components together within sections
            for section in self.sections:
                section.patch()
        else:
            # XXX from above: this won't happen unless something seriously breaks in the computer.
            print("Sorry, versions other than ? are not supported.")

    def run(self):
        """
        Runs Synth.
        Sets output to zero, then runs each component before extracting output
        from the final section (output_module).
        """
        self.out[:] = numpy.zeros(self.parameters["N_SAMP"])
        # Clear inputs and outputs in each component
        for section in self.sections:
            for component in section.components:
                component.clean()
        for section in self.sections:
            section.run()
        self.out[:] = self.output_module.out[:]

    def _get_16_16K(self):
        """
        Transforms output waveform to form for playing/saving.
        """
        assert self.parameters["FS"] == 10_000
        y = resample_poly(self.output, 8, 5)  # resample from 10K to 16K
        maxabs = numpy.max(numpy.abs(y))
        if maxabs > 1:
            y /= maxabs
        y = numpy.round(y * 32767).astype(numpy.int16)
        return y

    def play(self):
        """
        Plays output waveform.
        """
        y = self._get_int16at16K()
        simpleaudio.play_buffer(y, num_channels=1, bytes_per_sample=2, sample_rate=16_000)

    def save(self, path):
        """
        Saves output waveform to disk.
        Arguments:
            path (str): where the file should be saved
        """
        y = self._get_16_16K()
        write(path, 16_000, y)

class Section:
    '''
    Parent class for sections.
    NOTE: arguments.
    @param mast (Synth): Master Synth object, allows all sub-components to access params directly
    NOTE: attributes.
    @param mast (Synth): see Arguments
    @ins (list): list of Buffer objects for handling this Section's inputs, if it has any
    @outs (list): list of Buffer objects for handling this Section's outputs, if it has any
    Methods:
        connect (void): Connects sections
        process_ins(void): Processes input buffers
        process_outs(void): Processes output buffers
        run(void): Calls self.do(), which processes the signal by calling components' methods as necessary
    NOTE
    An operational Section needs two custom methods to be implemented on top of the default methods provided by the class definition:
        1) patch(), which connects components and
        2) do(), which describes the run order for said components + their parameters.
    \NOTE
    '''

    def __init__(self, mast):
        self.mast = mast
        self.components = []
        self.ins = []
        self.outs = []

    def connect(self, sections):
        """
        Connects this section to another.
        @param sections (list): list of Section objects to be connected to this Section
        NOTE
        for Section in sections, this method appends a Buffer to Section's outs and another Buffer to Section's ins.
        + connects the two so that signals can propagate between them. 
        See the documentation for the Component level operations of connect, send, and receive to understand more about how signal propagation happens.
        \NOTE
        """
        for section in sections:
            section.ins.append(Buffer(mast=self.mast))
            self.outs.append(Buffer(mast=self.mast, dests=[section.ins[-1]]))

    def process_ins(self):
        """
        Processes input buffers.
        """
        for _in in self.ins:
            _in.process()

    def process_outs(self):
        """
        Processes output buffers.
        """
        for out in self.outs:
            out.process()

    def run(self):
        """
        Carries out processing of this Section.
        """
        if self.ins is not None:
            self.process_ins()
        self.do()
        if self.outs is not None:
            self.process_outs()

class Component:
    """
    Parent class for components.
    NOTE: arguments.
        @param mast (Synth): master Synth object, allows for access to params
        @param dests (list): list of other Components, see send() method doc string for more information on how this list is used
    NOTE: attributes.
        @param mast (Synth): see Arguments
        @param dests (list): see Arguments
        @param input (Numpy array): input vector, length N_SAMP
        @param output (Numpy array): output vector, length N_SAMP
    Methods:
        receive (void): Changes input vector
        send (void): Propagates output to all destinations
        connect (void): Used to connect two components together
    NOTE
    Components are small signal processing units (e.g., filters, amplifiers) which compose Sections. 
    Components are connected at the Section level using the Components' connect() method. Components use the send() and receive() methods to send the signal down the chain.
    Components should all have a custom method implemented in which, at the very least, some output is put in the output attribute and, at the end of processing, /
        the send() method is called.
    \NOTE
    """
    def __init__(self, mast, dests=None):
        self.mast = mast
        if dests is None:
            self.dests = []
        else:
            self.dests = dests
        self.input = numpy.zeros(self.mast.parameters["N_SAMP"])
        self.output = numpy.zeros(self.mast.parameters["N_SAMP"])

    def receive(self, signal):
        """
        Updates current signal.
        @param signal (NumPy array): vector
        """
        self.input[:] = signal[:]

    def send(self):
        """
        Perpetuates signal to components further down in the chain.
        For each Component in this Component's dests list, uses the receive() method to set that Component's input to this Component's output, propagating the signal /
        through the components in a chain order.
        NOTE: Mixer has a custom implementation of receive, but it interfaces the same way.
        """
        for dest in self.dests:
            dest.receive(signal=self.output[:])

    def connect(self, components):
        """
        Connects two components together.
        Arguments:
            components (list): list of Components to be connected to
        For each destination Component in the list components, adds the
        destination Component to this Component's dests.
        """
        for component in components:
            self.dests.append(component)

    def clean(self):
        self.input = numpy.zeros(self.mast.parameters["N_SAMP"])
        self.output = numpy.zeros(self.mast.parameters["N_SAMP"])

