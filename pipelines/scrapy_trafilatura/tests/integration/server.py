from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import pymupdf

ARTICLE = ('Người bệnh cần điều trị và theo dõi với bác sĩ. Bệnh nhân dùng 0.5 mg/kg, HbA1c, SpO₂, H. pylori. ')*15
HTML = f'<html><head><title>Y khoa</title></head><body><article><h1>Điều trị</h1><p>{ARTICLE}</p><h2>Theo dõi</h2><p>{ARTICLE}</p></article></body></html>'.encode()


@contextmanager
def mock_server():
    counts = Counter()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split('?')[0]
            counts[route] += 1
            status, mime, body = 200, 'text/html; charset=utf-8', HTML
            extra = {}
            if route == '/robots.txt':
                mime, body = 'text/plain', b'User-agent: *\nDisallow: /robots-blocked\n'
            elif route == '/not-found':
                status, body = 404, b'not found'
            elif route == '/forbidden':
                status, body = 403, b'forbidden'
            elif route == '/rate-limit':
                status = 429 if counts[route] == 1 else 200
                extra['Retry-After'] = '1'
            elif route == '/server-error':
                status, body = 503, b'unavailable'
            elif route == '/redirect':
                status, body, extra['Location'] = 302, b'', '/ok-html'
            elif route == '/redirect-self':
                status, body, extra['Location'] = 302, b'', '/redirect-self'
            elif route in ('/redirect-a', '/redirect-b'):
                status, body, extra['Location'] = 302, b'', '/redirect-b' if route == '/redirect-a' else '/redirect-a'
            elif route == '/refresh-self':
                body = b'<html><head><meta http-equiv="refresh" content="0;url=/refresh-self"></head><body>refresh</body></html>'
            elif route == '/slow':
                time.sleep(1.2)
            elif route == '/empty':
                body = b''
            elif route == '/large':
                body = b'x'*150000
            elif route == '/wrong-header-html':
                mime = 'application/octet-stream'
            elif route == '/pdf':
                document = pymupdf.open()
                document.new_page().insert_text((72,72), 'Biomedical patient treatment 0.5 mg/kg HbA1c')
                body, mime = document.tobytes(), 'application/pdf'
                document.close()
            try:
                self.send_response(status)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(body)))
                for key, value in extra.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return  # Expected for the intentional timeout/size-abort fixture.
        def log_message(self, *args):
            return
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', counts
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
