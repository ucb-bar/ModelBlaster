"""A health-only stand-in for the LLM relay on my own node (probe2.sh asks /v1/health): the stub-LLM test never sends a
request; no key, no real endpoint."""
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    do_POST = do_GET
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
