from pydub import AudioSegment

def convert_mp3_to_wav(mp3_filename, wav_filename):
    audio = AudioSegment.from_mp3(mp3_filename)
    audio.export(wav_filename, format="wav")

mp3_filename = "CA1HA_87.mp3"
wav_filename = "CA1HA87.wav"
convert_mp3_to_wav(mp3_filename, wav_filename)
