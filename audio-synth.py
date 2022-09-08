from synthesizer import Player, Synthesizer, Waveform
import formantSynth

player = Player()
player.open_stream()
synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume = 1.0, use_osc2=False)


#Play major 3rds starting from A4:
player.play_wave(synthesizer.generate_constant_wave(440.0,1.0))
player.play_wave(synthesizer.generate_constant_wave(554.37,1.0))
player.play_wave(synthesizer.generate_constant_wave(659.25,1.0))

s=formantSynth.make()
s.run()
s.play()

