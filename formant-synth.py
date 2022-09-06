'''
formant-synth.py

Provides functions and formula calculations for producing formant speech synthesis.
Algorithm inspired by the 1980 Klatt Synthesizer, using similar input and variable values.


Author: Joey Ross
'''

'''
NOTE ABOUT DOCSTRINGS
----------------------------------
I prefer to use Javadoc when I can. However, Python doesn't have an equivalent (to my knowledge).
So I use some aspects of Javadoc in my comments to describe parameters and return values.
I know it doesn't do anything when compiled, but it's more what I'm used to when writing really large object-oriented programs.
Thank you for understanding my preference.
'''

'''
CLASSES
Param - Object containing parameters for synthesizer
Synth - synthesizer object, top-level
Section - section-level object, used for groups of parts of the synthesizer - cascade filters, voicing, etc - multiple Components in a row, essentially.
Component - Individual object for use in synthesizer, filter, amp, etc.
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

'''
NOTE ABOUT SECTIONS AND COMPONENTS
--------------------------------------
All of these are highly technical audio devices, which are more fit for an audio engineer.
While I have an understanding of how these devices work, some of the terms and descriptions are drawn from the various sources described in the /
    Independent Study study plan. 
Specifically, Resonator, Impulse, Mixer, Amplifier, and the various filters are very specifically engineered to fit this algorithm, and are drawn from the academic work of others. 
Great science is built on the shoulders of giants, right?
The sections are mostly self-authored but the components are a little bit more based on audio science research and less on CS/algorithms/data structures.
'''

class Voice(Section):
    """
    Generates a voicing waveform.
    Passes an impulse train with time-varying F0 through a series of filters to generate both normal and quasi-sinusoidal voicing waveforms. 
    Then, amplifies and mixes the two waveforms. Passes the mixed output onward
    through a time-varying binary switch.
    NOTE arguments:
    @param mast (Synth): see parent class
    NOTE attributes:
    @param impulse (Impulse): Periodic pulse train generator with fundamental frequency F0
    @param rg1 (Resonator): Glottal pole resonator to generate normal voicing waveform with center frequency FGR1 and bandwidth BGR1
    @param rgz (Resonator): Glottal zero antiresonator with center frequency FGZ bandwidth BGZ
    @param rg2 (Resonator): Secondary glottal resonator to generate quasi-sinusoidal voicing waveform with center frequency FGP and bandwidth BGR2
    @param av (Amplifier): Amplifier to control amplitude of normal voicing with amplification amount AV
    @param aqsv (Amplifier): Amplifier to control amplitude of quasi-sinuosidal voicing with amplification amount AQSV
    @param mixer (Mixer): Mixer to mix normal voicing waveform and quasi-sinusoidal voicing waveforms
    @param switch (Switch): Switch to switch destination of Voice to cascade filter track (CS=0) or parallel filter track with (CS=1)
    """
    def __init__(self, mast):
        Section.__init__(self, mast)
        self.impulse = Impulse(mast=self.mast)
        self.rg1 = Resonator(mast=self.mast)
        self.rgz = Resonator(mast=self.mast, anti=True)
        self.rg2 = Resonator(mast=self.mast)
        self.av = Amplifier(mast=self.mast)
        self.aqsv = Amplifier(mast=self.mast)
        self.mixer = Mixer(mast=self.mast)
        self.switch = Switch(mast=self.mast)
        self.components = [self.impulse, self.rg1, self.rgz, self.rg2, self.av, self.aqsv, self.mixer, self.switch]

    def patch(self):
        self.impulse.connect([self.rg1])
        self.rg1.connect([self.rgz, self.rg2])
        self.rgz.connect([self.av])
        self.rg2.connect([self.aqsv])
        self.av.connect([self.mixer])
        self.aqsv.connect([self.mixer])
        self.mixer.connect([self.switch])
        self.switch.connect([*self.outs])

    def do(self):
        self.impulse.impulse_gen(F0=self.mast.parameters["F0"])
        self.rg1.resonate(ff=self.mast.parameters["FGR1"],
                          bw=self.mast.parameters["BGR1"])
        self.rgz.resonate(ff=self.mast.parameters["FGZ"],
                          bw=self.mast.parameters["BGZ"])
        self.rg2.resonate(ff=self.mast.parameters["FGR1"],
                          bw=self.mast.parameters["BGR2"])
        self.av.amplify(dB=self.mast.parameters["AV"])
        self.aqsv.amplify(dB=self.mast.parameters["AQSV"])
        self.mixer.mix()
        self.switch.operate(choice=self.mast.params["CS"])


class Noise(Section):
    """
    Generates Gaussian noise.
    NOTE arguments:
    @param mast (KlattSynth): see parent class
    NOTE Attributes:
    @param noisegen (Noisegen): Gaussian noise generator
    @param lowpass (Lowpass): Simple time-domain lowpass filter
    @param amp (Amplifier): Amplifier to control amplitude of lowpassed noise
    """
    def __init__(self, mast):
        Section.__init__(self, mast)
        self.noisegen = Noisegen(mast=self.mast)
        self.lowpass = Lowpass(mast=self.mast)
        self.amp = Amplifier(mast=self.mast)
        self.components = [self.noisegen, self.lowpass, self.amp]

    def patch(self):
        self.noisegen.connect([self.lowpass])
        self.lowpass.connect([self.amp])
        self.amp.connect([*self.outs])

    def do(self):
        self.noisegen.generate()
        self.lowpass.filter()
        self.amp.amplify(dB=-60)  #TODO: real value?


class Cascade(Section):
    """
    Simulates a vocal tract using a cascade of resonators.
    NOTE Arguments:
    @param mast (KlattSynth): see parent class
    NOTE Attributes:
    @param aa (Amplifier): Control amplitude of noise waveform with amplification amount AH
    @param mixer (Mixer): mix voicing and noise
    @param rnp (Resonator): create nasal pole with center frequency FNP and bandwidth BNP
    @param rnz (Resonator): Antiresonator to create nasal zero with center frequency FNZ and bandwdith BNZ
    @param formants (list): List of Resonators, contains (N_FORM)x formants.
    """
    def __init__(self, mast):
        Section.__init__(self, mast)
        self.aa = Amplifier(mast=self.mast)
        self.mixer = Mixer(mast=self.mast)
        self.rnp = Resonator(mast=self.mast)
        self.rnz = Resonator(mast=self.mast, anti=True)
        self.formants = []
        for form in range(self.mast.parameters["N_FORM"]):
            self.formants.append(Resonator(mast=self.mast))
        self.components = [self.aa, self.mixer, self.rnp, self.rnz] + \
            self.formants

    def patch(self):
        self.ins[0].connect([self.mixer])
        self.ins[1].connect([self.aa])
        self.ah.connect([self.mixer])
        self.mixer.connect([self.rnp])
        self.rnp.connect([self.rnz])
        self.rnz.connect([self.formants[0]])
        for i in range(0, self.mast.parameters["N_FORM"]-1):
            self.formants[i].connect([self.formants[i+1]])
        self.formants[self.mast.parameters["N_FORM"]-1].connect([*self.outs])

    def do(self):
        self.aa.amplify(dB=self.mast.parameters["AH"])
        self.mixer.mix()
        self.rnp.resonate(ff=self.mast.parameters["FNP"],
                          bw=self.mast.parameters["BNP"])
        self.rnz.resonate(ff=self.mast.parameters["FNZ"],
                          bw=self.mast.parameters["BNZ"])
        for form in range(len(self.formants)):
            self.formants[form].resonate(ff=self.mast.parameters["FF"][form],
                                         bw=self.mast.parameters["BW"][form])


class KlattParallel1980(Section):
    """
    Simulates a vocal tract with a bank of parallel resonators.
    Directs the noise waveform to an amplifier, and the voicing waveform to a
    differentiator (highpass filter). Passes the high-passed voicing waveform
    and amplified noise waveform to amplifier-resonator pairs which correspond
    to the nasal formant and to formants 2-4. Passses the un-altered voicing
    waveform to an amplifier-resonator pair which corresponds to formant 1.
    Passes the amplified noise waveform to amplifier-resonator pairs which
    correspond to formants 5-6 and to a bypass path. Mixes the output of all
    the resonators and the bypass path.
    Arguments:
        mast (KlattSynth): see parent class
    Attributes:
        af (Amplifier): Amplifier to control the amplitude of the noise
            waveform (ins[1]) with amplification amount AF
        first_diff (Firstdiff): First differentiator (highpass filter)
        mixer (Mixer): Mixer to mix amplified noise waveform and highpass
            filtered voicing waveform
        an (Amplifier): Amplifier to control the amplitude of the nasal formant
            with amplification amount AN
        rnp (Resonator): Resonator to create the nasal formant, with center
            frequency FNP and bandwidth BNP
        a1 (Amplifier): Amplifier to control the amplitude of the first formant
            with amplification amount A1
        r1 (Resonator): Resonator to create the first formant, with center
            frequency and bandwidth in the 0-th array of FF and BW respectively
        a2 (Amplifier): Amplifier to control the amplitude of the first formant
            with amplification amount A2
        r2 (Resonator): Resonator to create the second formant, with center
            frequency and bandwidth in the 1-th array of FF and BW respectively
        a3 (Amplifier): Amplifier to control the amplitude of the second formant
            with amplification amount A3
        r3 (Resonator): Resonator to create the third formant, with center
            frequency and bandwidth in the 2-th array of FF and BW respectively
        a4 (Amplifier): Amplifier to control the amplitude of the third formant
            with amplification amount A4
        r4 (Resonator): Resonator to create the fourth formant, with center
            frequency and bandwidth in the 3-th array of FF and BW respectively
        a5 (Amplifier): Amplifier to control the amplitude of the fifth formant
            with amplification amount A5
        r5 (Resonator): Resonator to create the fifth formant, with center
            frequency and bandwidth in the 4-th array of FF and BW respectively
        a6 (Amplifier): Amplifier to control the amplitude of the sixth formant
            with amplification amount A6
        r6 (Resonator): Resonator to create the sixth formant, with center
            frequency and bandwidth in the 5-th array of FF and BW respectively
        ab (Amplifier): Amplifier to control the amplitude of the bypass path
            with amplification amount ?? (see comments below)
        output_mixer (Mixer): Mixer to mix various formant waveforms and bypass
            path output
    """
    def __init__(self, mast):
        KlattSection.__init__(self, mast)
        self.af = Amplifier(mast=self.mast)
        self.a1 = Amplifier(mast=self.mast)
        self.r1 = Resonator(mast=self.mast)
        self.first_diff = Firstdiff(mast=self.mast)
        self.mixer = Mixer(mast=self.mast)
        self.an = Amplifier(mast=self.mast)
        self.rnp = Resonator(mast=self.mast)
        self.a2 = Amplifier(mast=self.mast)
        self.r2 = Resonator(mast=self.mast)
        self.a3 = Amplifier(mast=self.mast)
        self.r3 = Resonator(mast=self.mast)
        self.a4 = Amplifier(mast=self.mast)
        self.r4 = Resonator(mast=self.mast)
        self.a5 = Amplifier(mast=self.mast)
        self.r5 = Resonator(mast=self.mast)
        # TODO: 6th formant currently not part of self.do()! Not sure what values
        # to give to it... need to keep reading Klatt 1980.
        self.a6 = Amplifier(mast=self.mast)
        self.r6 = Resonator(mast=self.mast)
        # TODO: ab currently not part of self.do()! Not sure what values to give
        # to it... need to keep reading Klatt 1980.
        self.ab = Amplifier(mast=self.mast)
        self.output_mixer = Mixer(mast=self.mast)
        self.components = [self.af, self.a1, self.r1, self.first_diff, \
                           self.mixer, self.an, self.rnp, self.a2, self.r2, \
                           self.r1, self.first_diff, self.mixer, self.an, \
                           self.rnp, self.a2, self.r2, self.a3, self.r3, \
                           self.a4, self.r4, self.a5, self.r5, self.a6, \
                           self.r6, self.ab, self.output_mixer]

    def patch(self):
        self.ins[1].connect([self.af])
        self.ins[0].connect([self.a1, self.first_diff])
        self.af.connect([self.mixer, self.a5, self.a6, self.ab])
        self.first_diff.connect([self.mixer])
        self.mixer.connect([self.an, self.a2, self.a3, self.a4])
        self.a1.connect([self.r1])
        self.an.connect([self.rnp])
        self.a2.connect([self.r2])
        self.a3.connect([self.r3])
        self.a4.connect([self.r4])
        self.a5.connect([self.r5])
        self.r6.connect([self.r6])
        for item in [self.r1, self.r2, self.r3, self.r4, self.r5, \
                     self.r6, self.rnp, self.ab]:
            item.connect([self.output_mixer])
        self.output_mixer.connect([*self.outs])

    def do(self):
        self.af.amplify(dB=self.mast.params["AF"])
        self.a1.amplify(dB=self.mast.params["A1"])
        self.r1.resonate(ff=self.mast.params["FF"][0],
                         bw=self.mast.params["BW"][0])
        self.first_diff.differentiate()
        self.mixer.mix()
        self.an.amplify(dB=self.mast.params["AN"])
        self.rnp.resonate(ff=self.mast.params["FNP"],
                          bw=self.mast.params["BNP"])
        self.a2.amplify(dB=self.mast.params["A2"])
        self.r2.resonate(ff=self.mast.params["FF"][1],
                         bw=self.mast.params["BW"][1])
        self.a3.amplify(dB=self.mast.params["A3"])
        self.r3.resonate(ff=self.mast.params["FF"][2],
                         bw=self.mast.params["BW"][2])
        self.a4.amplify(dB=self.mast.params["A4"])
        self.r4.resonate(ff=self.mast.params["FF"][3],
                         bw=self.mast.params["BW"][3])
        self.a5.amplify(dB=self.mast.params["A5"])
        self.r5.resonate(ff=self.mast.params["FF"][4],
                         bw=self.mast.params["BW"][4])
        self.output_mixer.mix()


class KlattRadiation1980(KlattSection):
    """
    Simulates the effect of radiation characteristic in the vocal tract.
    Simply mixes inputs, and then highpass filters them (via calculating
    the first derivative).
    Arguments:
        mast (KlattSynth): see parent class
    Attributes:
        mixer (Mixer): Mixer to mix various inputs
        firstdiff (Firstdiff): First differentiator to act as highpass filter
            (in this case, models the effect of the radiation characteristic of
            the lips)
    """
    def __init__(self, mast):
        KlattSection.__init__(self, mast)
        self.mixer = Mixer(mast=self.mast)
        self.firstdiff = Firstdiff(mast=self.mast)
        self.components = [self.mixer, self.firstdiff]

    def patch(self):
        for _in in self.ins:
            _in.connect([self.mixer])
        self.mixer.connect([self.firstdiff])
        self.firstdiff.connect([*self.outs])

    def do(self):
        self.mixer.mix()
        self.firstdiff.differentiate()


class OutputModule(KlattSection):
    """
    Mixes inputs and then normalizes mixed waveform by setting peak value of 1.
    Arguments:
        mast (KlattSynth): see parent class
    Attributes:
        mixer (Mixer): Mixer to mix various inputs
        normalizer (Normalizer): Divides waveform by its absolute value maximum
        output (np.array): Final destination for synthesized speech waveform,
            extracted by KlattSynth object after synthesis is complete
    """
    def __init__(self, mast):
        KlattSection.__init__(self, mast)
        self.mixer = Mixer(mast=self.mast)
        self.normalizer = Normalizer(mast=self.mast)
        self.output = np.zeros(self.mast.params["N_SAMP"])
        self.components = [self.mixer, self.normalizer]

    def patch(self):
        for _in in self.ins:
            _in.dests = [self.mixer]
        self.mixer.dests = [self.normalizer]
        self.normalizer.dests = [*self.outs]

    def do(self):
        self.mixer.mix()
        self.normalizer.normalize()
        self.output[:] = self.normalizer.output[:]


##### COMPONENT DEFINITIONS #####
class Buffer(KlattComponent):
    """
    Utility component used in signal propagation.
    Arguments:
        mast (KlattSynth): see parent class
        dests (None): see parent class
    """
    def __init__(self, mast, dests=None):
        KlattComponent.__init__(self, mast, dests)

    def process(self):
        """
        Sets output to be the input waveform, and then sends output waveform to
        downstream connected components.
        """
        self.output[:] = self.input[:]
        self.send()


class Resonator(KlattComponent):
    """
    Klatt resonator.
    Recursive time-domain implementation of a resonator, matching Klatt's
    specification.
    Arguments:
        mast (KlattSynth): see parent class
        anti (boolean): determines whether Resonator acts as resonator or
            antiresonator (for more information, see Klatt 1980)
    Attributes:
        anti (boolean): See Arguments
    """
    def __init__(self, mast, anti=False):
        KlattComponent.__init__(self, mast)
        self.anti = anti

    def calc_coef(self, ff, bw):
        """
        Calculates filter coefficients.
        Calculates filter coefficients to implement resonator ala Klatt 1980.
        If self.anti = True, modifies the coefficients after calculation to
        turn the resonator into an antiresonator. Accepts center frequency and
        bandwidth values, and accesseds non-time-varying parameters from mast.
        Arguments:
            ff (array): Array of center frequency values in Hz, with length
                N_SAMP
            bw (array): Array of bandwidth values in Hz, with length N_SAMP
        """
        c = -np.exp(-2*np.pi*bw*self.mast.params["DT"])
        b = (2*np.exp(-np.pi*bw*self.mast.params["DT"])\
             *np.cos(2*np.pi*ff*self.mast.params["DT"]))
        a = 1-b-c
        if self.anti:
            a_prime = 1/a
            b_prime = -b/a
            c_prime = -c/a
            return(a_prime, b_prime, c_prime)
        else:
            return(a, b, c)

    def resonate(self, ff, bw):
        """
        Processes input waveform with resonator filter.
        Loops through values in the input array, calculating filter outputs
        sample-by-sample in the time domain. Takes arrays to indicate center
        frequency and bandwidth values, and passes them to calc_coef() to get
        coefficients to be used in the filtering calculation.
        Arguments:
            ff (array): Array of center frequency values in Hz, with length
                N_SAMP
            bw (array): Array of bandwidth values in Hz, with length N_SAMP
        """
        a, b, c = self.calc_coef(ff, bw)
        self.output[0] = a[0]*self.input[0]
        if self.anti:
            self.output[1] = a[1]*self.input[1] + b[1]*self.input[0]
            for n in range(2, self.mast.params["N_SAMP"]):
                self.output[n] = a[n]*self.input[n] + b[n]*self.input[n-1] \
                                + c[n]*self.input[n-2]
        else:
            self.output[1] = a[1]*self.input[1] + b[1]*self.output[0]
            for n in range(2,self.mast.params["N_SAMP"]):
                self.output[n] = a[n]*self.input[n] + b[n]*self.output[n-1] \
                                + c[n]*self.output[n-2]
        self.send()


class Impulse(KlattComponent):
    """
    Time-varying impulse generator.
    (NOTE: Impulse train ideas + math drawn from: https://ccrma.stanford.edu/~jos/sasp/Impulse_Trains.html
    Arguments:
        mast (KlattSynth): see parent class
    Attributes:
        last_glot_pulse (int): Number of samples since last glottal pulse
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)
        self.last_glot_pulse = 0

    def impulse_gen(self, F0):
        """
        Generates impulse train.
        Starts with array of zeros with length N_SAMP. Loops through array,
        setting value to 1 when the time since last glottal pulse is equal
        to or exceeds the current glotal period (inverse of current F0).
        Arguments:
            F0 (arrry): Array of F0 values at each sample
        """
        glot_period = np.round(self.mast.params["FS"]/F0)
        self.last_glot_pulse = 0
        for n in range(self.mast.params["N_SAMP"]):
            if n - self.last_glot_pulse >= glot_period[n]:
                self.output[n] = 1
                self.last_glot_pulse = n
        self.send()


class Mixer(KlattComponent):
    """
    Mixes waveforms together.
    Arguments:
        mast (KlattSynth): see parent class
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)

    def receive(self, signal):
        """
        Mixes incoming waveform with current input.
        Replaces KlattComponent's receive() method. Instead of setting input
        equal to incoming waveform, mixes input with incoming waveform.
        Arguments:
            signal (array): waveform to be mixed with input
        """
        self.input[:] = self.input[:] + signal[:]

    def mix(self):
        """
        Sets output to input.
        The above receive() method really does the mixing --- this is just the
        method called by the Mixer's KlattSection so that the signal
        propagates.
        """
        self.output[:] = self.input[:]
        self.send()


class Amplifier(KlattComponent):
    """
    Simple amplifier, scales amplitude of signal by dB value.
    Arguments:
        mast (KlattSynth): see parent class
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)

    def amplify(self, dB):
        """
        Peforms amplification.
        Arguments:
            dB (float): amount of amplification to occur in dB
        """
        dB = np.sqrt(10)**(dB/10)
        self.output[:] = self.input[:]*dB
        self.send()


class Firstdiff(KlattComponent):
    """
    Simple first difference operator.
    Arguments:
        mast (KlattSynth): see parent class
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)

    def differentiate(self):
        """
        Peforms first difference operation.
        """
        self.output[0] = 0
        for n in range(1, self.mast.params["N_SAMP"]):
            self.output[n] = self.input[n] - self.input[n-1]
        self.send()


class Lowpass(KlattComponent):
    """
    Simple one-zero 6 dB/oct lowpass filter.
    Arguments:
        mast (KlattSynth): see parent class
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)

    def filter(self):
        """
        Implements lowpass filter operation.
        """
        self.output[0] = self.input[0]
        for n in range(1, self.mast.params["N_SAMP"]):
            self.output[n] = self.input[n] + self.output[n-1]
        self.send()


class Normalizer(KlattComponent):
    """
    Normalizes signal so that abs(max value) is 1.
    Arguments:
        mast (KlattSynth): see parent class
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)

    def normalize(self):
        """
        Implements normalization.
        """
        self.output[:] = self.input[:]/np.max(np.abs(self.input[:]))
        self.send()


class Noisegen(KlattComponent):
    """
    Generates noise from a Gaussian distribution.
    Arguments:
        mast (KlattSynth): see parent class
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)

    def generate(self):
        """
        Generates Gaussian noise with mean 0, sd 1.0, and length N_SAMP
        """
        self.output[:] = np.random.normal(loc=0.0, scale=1.0,
                                          size=self.mast.params["N_SAMP"])
        self.send()


class Switch(KlattComponent):
    """
    Binary switch between two outputs.
    Has two output signals (instead of one, as in other KlattComponents). Each
    is connected to a different destination, and the operate() function
    switches the input between the two possible outputs depending on a control
    singal.
    Arguments:
        mast (KlattSynth): see parent class
    Attributes:
        output (list): List of two np.arrays as described above
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)
        self.output = []
        self.output.append(np.zeros(self.mast.params["N_SAMP"]))
        self.output.append(np.zeros(self.mast.params["N_SAMP"]))

    def send(self):
        """
        Perpetuates signal to components further down in the chain.
        Replaces KlattComponent's send() method, sending one output to one
        destination and the other output to another destination.
        """
        self.dests[0].receive(signal=self.output[0][:])
        self.dests[1].receive(signal=self.output[1][:])

    def operate(self, choice):
        """
        Implements binary switching.
        Arguments:
            choice (np.array): Array of zeros and ones which tell the Switch
                where to send the input singal. For samples where switch=0 the
                signal is sent to the first output and the second output is set
                to zero. For samples where switch=1 the signal is sent to the
                second output and the first output is set to zero.
        """
        for n in range(self.mast.params["N_SAMP"]):
            if choice[n] == 0:
                self.output[0][n] = self.input[n]
                self.output[1][n] = 0
            elif choice[n] == 1:
                self.output[0][n] = 0
                self.output[1][n] = self.input[n]
        self.send()

    def clean(self):
        self.output = []
        self.output.append(np.zeros(self.mast.params["N_SAMP"]))
        self.output.append(np.zeros(self.mast.params["N_SAMP"]))


if __name__ == '__main__':
    s = klatt_make(KlattParam1980(DUR=0.5)) # Creates a Klatt synthesizer w/ default settings
    # see also: http://www.fon.hum.uva.nl/david/ma_ssp/doc/Klatt-1980-JAS000971.pdf
    N = s.params["N_SAMP"]
    F0 = s.params["F0"]
    FF = np.asarray(s.params["FF"]).T
    AV = s.params["AV"]
    AH = s.params['AH']

    # amplitude / voicing
    AV[:] = np.linspace(1, 0, N) ** 0.1 * 60
    if 1:  # unvoiced consonant
        Nv1 = 800  # start of unvoiced-voiced transition
        Nv2 = 1000  # end of unvoiced-voiced transition
        AV[:Nv1] = 0
        AH[:Nv1] = 55
        AV[Nv1:Nv2] = np.linspace(0, AV[Nv2], Nv2-Nv1)
        AH[Nv1:Nv2] = np.linspace(55, 0, Nv2-Nv1)


    # F0
    F0[:] = np.linspace(120, 70, N)  # a falling F0 contour

    # FF
    target1 = np.r_[300, 1000, 2600]  # /b/
    #target2 = np.r_[280, 2250, 2750]  # /i/
    target2 = np.r_[750, 1300, 2600]  # /A/
    if 0:  # linear transition
        xfade = np.linspace(1, 0, N)
    else:  # exponential transition
        n = np.arange(N)
        scaler = 20
        xfade = 2 / (1 + np.exp(scaler * n / (N-1)))
    FF[:,:3] = np.outer(xfade, target1) + np.outer((1 - xfade), target2)

    # synthesize
    s.params["FF"] = FF.T
    s.run()
    s.play()
    s.save('synth.wav')

    # visualize
    t = np.arange(len(s.output)) / s.params['FS']
    import matplotlib.pyplot as plt
    ax = plt.subplot(211)
    plt.plot(t, s.output)
    plt.axis(ymin=-1, ymax=1)
    plt.ylabel('amplitude')
    plt.twinx()
    plt.plot(t, AV, 'r', label='AV')
    plt.plot(t, AH, 'g', label='AH')
    plt.legend()
    plt.subplot(212, sharex=ax)
    plt.specgram(s.output, Fs=s.params['FS'])
    plt.plot(t, FF, alpha=0.5)
    plt.xlabel('time [s]')
    plt.ylabel('frequency [Hz]')
    plt.savefig('figure.pdf')
    plt.show()