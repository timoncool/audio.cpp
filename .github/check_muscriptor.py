import array
import base64
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import wave

label=sys.argv[1]
root=Path("fixtures").resolve()
root.mkdir(exist_ok=True)
model=root/"muscriptor-small"
model.mkdir(exist_ok=True)
for name,sha in {"config.json":"3008fc481e4a1cd978e337eb3759260c270892204db5039235ac939e1f42aeb2",
                 "model.safetensors":"bbd482c786b895cf7d8f44185073d951adae2ebb8a66f82ca84cd1f84569549c"}.items():
    path=model/name
    if not path.exists():
        subprocess.run(["curl","-fsSL","--retry","3","https://huggingface.co/cocktailpeanut/muscriptor-small/resolve/31a8f75d6a8b5383fd71ad1371dc1620389ab722/"+name,"-o",path],check=True)
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest()==sha
notes=[60,64,67,72]
rate=24000
# General MIDI acoustic grand, four quarter notes with gaps at 60 BPM.
import struct
track=bytearray.fromhex("00 ff 51 03 0f 42 40 00 c0 00")
for pitch in notes:
    track.extend((24,0x90,pitch,85,72,0x80,pitch,0))
track.extend((96,255,47,0))
mid=root/"piano.mid"
mid.write_bytes(b"MThd"+struct.pack(">IHHH",6,0,1,96)+b"MTrk"+struct.pack(">I",len(track))+track)
audio=root/"piano.wav"
subprocess.run(["fluidsynth","-ni","-g","0.7","-r","24000","-F",str(audio),"/usr/share/sounds/sf2/FluidR3_GM.sf2",str(mid)],check=True)
import collections
import torch
from muscriptor.events import NoteStartEvent, NoteEndEvent
from muscriptor.transcription_model import TranscriptionModel
torch.set_num_threads(4)
torch.manual_seed(0)
reference=TranscriptionModel.load_model(weights_path=model/"model.safetensors",device="cpu",dtype="float32")
def transcribe(wavfile):
    starts={}
    found=[]
    for event in reference.transcribe(audio=wavfile):
        if isinstance(event,NoteStartEvent):
            starts[event.index]=event
        elif isinstance(event,NoteEndEvent):
            note=starts[event.start_event_index]
            found.append((note.pitch,note.start_time,event.end_time,note.instrument))
    matches=[pitch for i,pitch in enumerate(notes) if any(n==pitch and abs(start-i-.25)<.25 and end-start>.2 for n,start,end,_ in found)]
    print(json.dumps(dict(clip=wavfile.name,input_sha256=hashlib.sha256(wavfile.read_bytes()).hexdigest(),count=len(found),
                          instruments=dict(collections.Counter(str(n[3]) for n in found)),matched=matches,first_notes=found[:12])),flush=True)
    return matches
grand=transcribe(audio)
samples=array.array("h",[0]*int(.25*rate))
for note in notes:
    hz=440*2**((note-69)/12)
    for frame in range(rate):
        t=frame/rate
        env=min(1,t/.012)*math.exp(-2*t)*max(0,min(1,(.8-t)/.08))
        value=sum(math.sin(2*math.pi*hz*t*k)*gain for k,gain in enumerate([0,1,.44,.21,.1,.04]))
        samples.append(round(13000*env*value))
samples.extend([0]*rate)
audio=root/"harmonics.wav"
with wave.open(str(audio),"wb") as stream:
    stream.setparams((1,2,rate,0,"NONE","not compressed"))
    stream.writeframes(samples.tobytes())
harmonics=transcribe(audio)
assert grand==notes,(grand,harmonics)
