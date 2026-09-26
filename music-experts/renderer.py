# Autor: Adam Skrodzki
"""Render MIDI -> WAV (sine-wave synthesis, no external binaries needed).
Usage: python renderer.py input.mid [output.wav]
"""
import sys
import numpy as np
import pretty_midi
import wave

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python renderer.py input.mid [output.wav]")
    mid_path = sys.argv[1]
    wav_path = sys.argv[2] if len(sys.argv) > 2 else mid_path.rsplit(".", 1)[0] + ".wav"

    pm = pretty_midi.PrettyMIDI(mid_path)
    instruments = pm.instruments
    note_counts = [len(i.notes) for i in instruments]
    duration = pm.get_end_time()
    print(f"in:   {mid_path}")
    print(f"      instruments: {len(instruments)} | notes per instrument: {note_counts} | duration: {duration:.1f}s")

    if sum(note_counts) == 0:
        sys.exit("ERROR: MIDI contains no notes — nothing to render. Check the generation/conversion step.")

    audio = pm.synthesize(fs=44100)
    peak = float(np.abs(audio).max())
    print(f"      peak amplitude before normalize: {peak:.4f}")
    if peak == 0:
        sys.exit("ERROR: synthesis produced silence despite notes present.")

    audio = (audio / peak * 32767).astype(np.int16)
    with wave.open(wav_path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(audio.tobytes())
    print(f"out:  {wav_path} | {len(audio) / 44100:.1f}s @ 44.1kHz mono int16")

if __name__ == "__main__":
    main()
