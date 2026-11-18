# Which models does this node's relay serve? One tiny request per model; prints status + usage only (no secrets).
. /tmp/krisd/farm/relay.env
. /tmp/krisd/venv/bin/activate
python - "$@" <<'PY'
import json, os, sys, time, urllib.request
url = os.environ["KB_LLM_SERVER_URL"] + "/v1/generate"
tok = open(os.environ["KB_LLM_SERVER_TOKEN_FILE"]).read().strip()
for m in sys.argv[1:]:
    body = json.dumps({"messages": [{"role": "user", "content": "Reply with the single word OK."}], "n_tasks": 1,
                       "model": m, "client": "bt-modelcheck", "role": "probe"}).encode()
    rq = urllib.request.Request(url, data=body, headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    t = time.time()
    try:
        r = json.loads(urllib.request.urlopen(rq, timeout=180).read())
        g = (r.get("generations") or [""])[0]
        u = r.get("usage") or {}
        print(m, "OK", f"{time.time()-t:.1f}s", repr(g[:40]), json.dumps({k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}),
              "model_echo=", r.get("model"))
    except urllib.error.HTTPError as e:
        print(m, "HTTP", e.code, e.read()[:300].decode(errors="replace"))
    except Exception as e:
        print(m, "ERR", repr(e)[:300])
PY
