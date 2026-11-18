# my node: fake health-only relay (no key, no endpoint) + the stub plugin dir. Needs /tmp/krisd/hw/{fake_relay.py,hw_plugin.py}.
mkdir -p /tmp/krisd/farm /tmp/krisd/hw
echo token > /tmp/krisd/hw/token
cat > /tmp/krisd/farm/relay.env <<'ENV'
export KB_LLM_SERVER_URL=http://127.0.0.1:18099
export KB_LLM_SERVER_TOKEN_FILE=/tmp/krisd/hw/token
ENV
pgrep -f "fake_relay.py 18099" >/dev/null || { nohup setsid /home/utils/Python-3.12.5/bin/python3 /tmp/krisd/hw/fake_relay.py 18099 </dev/null >/tmp/krisd/hw/relay.log 2>&1 & sleep 1; }
curl -sf -m 5 -H "Authorization: Bearer token" http://127.0.0.1:18099/v1/health && echo relay-ok
cat /tmp/krisd/farm/setup.status 2>/dev/null
