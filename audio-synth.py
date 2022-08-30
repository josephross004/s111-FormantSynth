from synthesizer import Player, Synthesizer, Waveform

player = Player()
player.open_stream()
synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume = 1.0, use_osc2=False)


'''
#Play major 3rds starting from A4:
player.play_wave(synthesizer.generate_constant_wave(440.0,1.0))
player.play_wave(synthesizer.generate_constant_wave(554.37,1.0))
player.play_wave(synthesizer.generate_constant_wave(659.25,1.0))
'''

player.play_wave(Synthesizer(osc1_waveform=Waveform.sine, osc1_volume = 0.5, use_osc2=False).generate_chord([270,2300,3000],1.0))
player.play_wave(Synthesizer(osc1_waveform=Waveform.sine, osc1_volume = 0.5, use_osc2=False).generate_chord([300,870,2250],1.0))
player.play_wave(Synthesizer(osc1_waveform=Waveform.sine, osc1_volume = 0.5, use_osc2=False).generate_chord([400,2000,2550],1.0))


