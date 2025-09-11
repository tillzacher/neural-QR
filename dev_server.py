import os, json
from http.server import SimpleHTTPRequestHandler, HTTPServer

IMAGES_DIR = "readme_imgs"

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/list"):
            root = os.path.join(os.getcwd(), IMAGES_DIR)
            files = []
            if os.path.isdir(root):
                for f in os.listdir(root):
                    p = os.path.join(root, f)
                    if os.path.isfile(p) and f.lower().split(".")[-1] in ("png","jpg","jpeg","webp"):
                        files.append(f)
            data = json.dumps({"files": files}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

if __name__ == "__main__":
    os.chdir(os.getcwd())
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
