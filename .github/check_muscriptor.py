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
samples=array.array("h",[0]*int(.25*rate))
for note in notes:
    hz=440*2**((note-69)/12)
    for frame in range(rate):
        t=frame/rate
        env=min(1,t/.012)*math.exp(-2*t)*max(0,min(1,(.8-t)/.08))
        value=sum(math.sin(2*math.pi*hz*t*k)*gain for k,gain in enumerate([0,1,.44,.21,.1,.04]))
        samples.append(round(13000*env*value))
samples.extend([0]*rate)
audio=root/"piano.wav"
with wave.open(str(audio),"wb") as stream:
    stream.setparams((1,2,rate,0,"NONE","not compressed"))
    stream.writeframes(samples.tobytes())
with socket.socket() as sock:
    sock.bind(("127.0.0.1",0))
    port=sock.getsockname()[1]
exe,=Path("released" if label=="released" else "candidate/build/qa").rglob("audiocpp_server")
cfg=root/(label+".json")
cfg.write_text(json.dumps(dict(host="127.0.0.1",port=port,threads=4,backend="cpu",lazy_load=True,
                               models=[dict(id="muscriptor-small",family="muscriptor",mode="offline",task="midi",path=str(model))])))
def call(path,body=None):
    req=Request(f"http://127.0.0.1:{port}"+path,data=json.dumps(body).encode() if body else None,headers={"Content-Type":"application/json"})
    try:
        with urlopen(req,timeout=600) as response: return json.load(response)
    except HTTPError as error:
        raise RuntimeError(error.code,error.read()) from error
proc=subprocess.Popen([str(exe.resolve()),"--config",str(cfg)],cwd=exe.parent)
try:
    for i in range(100):
        if proc.poll() is not None: raise RuntimeError(proc.returncode)
        try:
            health=call("/health")
            break
        except URLError: time.sleep(.25)
    else: raise TimeoutError("startup")
    assert health["backend"]=="cpu" and ("31cb1a4a" in str(health) if label=="released" else health["git"]=="41242418"),health
    body={"model":"muscriptor-small","stream_format":"sse","include_artifacts":True,"request":{"audio":str(audio)}}
    req=Request(f"http://127.0.0.1:{port}/v1/tasks/stream",data=json.dumps(body).encode(),
                headers={"Accept":"text/event-stream","Content-Type":"application/json"})
    streamed=[]
    result=None
    done=False
    with urlopen(req,timeout=600) as response:
        for raw in response:
            line=raw.decode().strip()
            if not line.startswith("data:"): continue
            value=line[5:].strip()
            if value=="[DONE]":
                done=True
                break
            data=json.loads(value)
            if data["type"]=="error": raise RuntimeError(data)
            if data["type"]=="task.stream.done": result=data["result"]
            for art in data.get("event",{}).get("artifacts",[]):
                if art.get("meta",{}).get("format")=="muscriptor-event-json":
                    streamed.append(json.loads(base64.b64decode(art["payload"])))
    assert done and result is not None
    midi,=[base64.b64decode(a["payload"]) for a in result["artifacts"] if a["kind"]=="midi"]
    events=json.loads(result["text"])
    print("Received MIDI bytes, incremental events, final events:",len(midi),len(streamed),len(events),flush=True)
    assert midi.startswith(b"MThd") and b"MTrk" in midi and streamed
    ends={e["start_event_index"]:e["end_time"] for e in events if e["type"]=="end"}
    found=[(e["pitch"],round(e["start_time"],3),round(ends[e["index"]],3),e["instrument"]) for e in events if e["type"]=="start" and e["index"] in ends]
    print(json.dumps(dict(label=label,health=health,notes=found)),flush=True)
    for i,pitch in enumerate(notes):
        assert any(p==pitch and abs(start-i-.25)<.25 and end-start>.20 for p,start,end,instrument in found),(pitch,found)
    (root/(label+"-notes.json")).write_text(json.dumps(found))
finally:
    if proc.poll() is None: proc.terminate()
    proc.wait(timeout=20)
