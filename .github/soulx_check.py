import array
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import wave

mode = sys.argv[1]
fixtures = Path("fixtures").resolve()
manifest = json.loads((Path(__file__).parent/f"soulx-{mode}.json").read_text())
entry, = manifest["files"]
url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{entry['path']}"
model = fixtures/Path(entry["path"]).name
subprocess.run(["curl", "-fsSL", "--retry","3",url,"-o",str(model)],check=True)
assert model.stat().st_size == entry["bytes"]
with model.open("rb") as f:
    assert hashlib.file_digest(f,"sha256").hexdigest() == entry["sha256"]
options = {"seed":"2026", "num_steps":"32"}
task = "gen" if mode=="svs" else "svc"
request = {"seed":2026,"options":options}
if mode=="svs":
    request["voice_ref"]=str(fixtures/"en_prompt.wav")
    options.update(reference_metadata_file=str(fixtures/"en_prompt.json"),target_metadata_file=str(fixtures/"en_target.json"),control="score")
    seconds=json.loads((fixtures/"en_target.json").read_text())[-1]["time"][1]/1000
else:
    request.update(audio=str(fixtures/"svc_target.wav"),voice_ref=str(fixtures/"svc_prompt.wav"))
    seconds=6
with socket.socket() as sock:
    sock.bind(("127.0.0.1",0))
    port=sock.getsockname()[1]
config={"host":"127.0.0.1","port":port,"threads":4,"backend":"cpu","lazy_load":True,
        "models":[{"id":"soulx_singer","family":"soulx_singer","mode":"offline","task":task,"path":str(model)}]}
cfg=fixtures/f"{mode}.json"
cfg.write_text(json.dumps(config))
def api(route,body=None):
    req=Request(f"http://127.0.0.1:{port}"+route,data=json.dumps(body).encode() if body else None,headers={"Content-Type":"application/json"})
    try:
        with urlopen(req,timeout=1800 if body else 5) as response: return json.load(response)
    except HTTPError as error:
        raise RuntimeError(error.code,error.read()) from error
exe, = Path("candidate/build/qa").rglob("audiocpp_server")
process=subprocess.Popen([str(exe.resolve()),"--config",str(cfg)],cwd="candidate")
try:
    for _ in range(200):
        if process.poll() is not None: raise RuntimeError(process.returncode)
        try:
            health=api("/health")
            break
        except URLError: time.sleep(.25)
    else: raise TimeoutError("server startup")
    assert health["git"]=="41242418" and health["backend"]=="cpu",health
    started=time.monotonic()
    result=api("/v1/tasks/run",{"model":"soulx_singer","request":request})
    elapsed=time.monotonic()-started
    wav=base64.b64decode(result.get("result",result)["audio"].split(",")[-1],validate=True)
    subprocess.run(["ffmpeg","-v","error","-i","pipe:0","-f","null","-"],input=wav,check=True)
    with wave.open(io.BytesIO(wav)) as f:
        assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(24000,1,2)
        duration=f.getnframes()/24000
        assert abs(duration-seconds)<.05,(duration,seconds)
        samples=array.array("h",f.readframes(f.getnframes()))
        rms=math.sqrt(sum(s*s for s in samples)/len(samples))
        assert rms>1,rms
    print(json.dumps(dict(mode=mode,seconds=duration,rms_pcm16=rms,elapsed=elapsed,health=api("/health"),wav_sha256=hashlib.sha256(wav).hexdigest())))
finally:
    process.terminate()
    process.wait(timeout=20)
