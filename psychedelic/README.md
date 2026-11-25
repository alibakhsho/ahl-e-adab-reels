# Psychedelic reel generator

Turns `source.jpg` into a 16 s, 1080x1920, 140 BPM beat-synced reel with an original synthesized psytrance track.

```
python3 audio.py track.wav
python3 render.py source.jpg track.wav ../output/psychedelic_reel.mp4
python3 render.py source.jpg track.wav preview.png 300   # single-frame preview
```

Needs numpy, Pillow and ffmpeg. Timeline (beats): 0-8 intro/build, 8-16 groove, 16-32 drop, 32-36 outro.
