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

family=sys.argv[1]
fixtures=Path("fixtures").resolve()
fixtures.mkdir(exist_ok=True)
manifest, = [item for item in json.loads((Path(__file__).parent/"music-weights.json").read_text(encoding="utf-8-sig"))
             if item["family"]==family]
entry, = manifest["files"]
url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{entry['path']}"
model = fixtures/Path(entry["path"]).name
subprocess.run(["curl", "-fsSL", "--retry","3",url,"-o",str(model)],check=True)
assert model.stat().st_size == entry["bytes"]
with model.open("rb") as f:
    assert hashlib.file_digest(f,"sha256").hexdigest() == entry["sha256"]
task="gen"
options={"seed":"2026"}
request={"seed":2026,"options":options}
if family=="magenta_realtime_2":
    request["text"]="Disco funk with a steady bass line and syncopated guitar."
    options.update(duration_sec="4")
elif family=="midashenglm_gen":
    request["text"]="<|caption|> Soft instrumental piano with gentle rain <|asr|> <|unknown|> <|speech|> <|unknown|> <|music|> warm sparse piano melody <|sfx|> distant rain <|env|> small room"
    options.update(duration_sec="4")
else:
    assert family=="vevo2"
    task="tts"
    ref=fixtures/"singer.wav"
    subprocess.run(["curl","-fsSL","--retry","3","https://raw.githubusercontent.com/Soul-AILab/SoulX-Singer/81aeb3ae772c70093c3de74dc23c92d983801ae4/example/audio/svc_prompt_demo.mp3","-o",str(fixtures/"reference.mp3")],check=True)
    subprocess.run(["ffmpeg","-v","error","-i",str(fixtures/"reference.mp3"),"-ar","24000","-ac","1",str(ref)],check=True)
    request.update(text="We follow the morning light",voice_ref=str(ref))
    options.update(route="text_to_singing",target_text=request["text"],max_tokens="256")
with socket.socket() as sock:
    sock.bind(("127.0.0.1",0))
    port=sock.getsockname()[1]
config={"host":"127.0.0.1","port":port,"threads":4,"backend":"cpu","lazy_load":True,
        "models":[{"id":family,"family":family,"mode":"offline","task":task,"path":str(model)}]}
cfg=fixtures/f"{family}.json"
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
    result=api("/v1/tasks/run",{"model":family,"request":request})
    elapsed=time.monotonic()-started
    wav=base64.b64decode(result.get("result",result)["audio"].split(",")[-1],validate=True)
    subprocess.run(["ffmpeg","-v","error","-i","pipe:0","-f","null","-"],input=wav,check=True)
    with wave.open(io.BytesIO(wav)) as f:
        hz,channels=f.getframerate(),f.getnchannels()
        assert 8000<=hz<=96000 and channels in (1,2) and f.getsampwidth()==2,(hz,channels)
        duration=f.getnframes()/hz
        assert duration>=0.1,duration
        if family=="magenta_realtime_2":
            assert (hz,channels)==(48000,2) and abs(duration-4)<.05,(hz,channels,duration)
        samples=array.array("h",f.readframes(f.getnframes()))
        rms=math.sqrt(sum(s*s for s in samples)/len(samples))
        assert rms>1,rms
    print(json.dumps(dict(family=family,rate=hz,channels=channels,seconds=duration,rms_pcm16=rms,elapsed=elapsed,health=api("/health"),wav_sha256=hashlib.sha256(wav).hexdigest())))
finally:
    process.terminate()
    process.wait(timeout=20)
    model.unlink()
