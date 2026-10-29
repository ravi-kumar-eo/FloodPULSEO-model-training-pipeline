"""Read members of a remote zip over HTTP range requests (curl)."""
import io, subprocess, zipfile
class RangeFile(io.RawIOBase):
    def __init__(s, url):
        s.url = url; s.pos = 0
        out = subprocess.run(['curl', '-sIL', url], capture_output=True, text=True).stdout
        s.size = int([l.split(':')[1] for l in out.splitlines() if l.lower().startswith('content-length')][-1])
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, o, w=0):
        s.pos = {0: o, 1: s.pos + o, 2: s.size + o}[w]; return s.pos
    def readinto(s, b):
        if s.pos >= s.size: return 0
        end = min(s.pos + len(b), s.size) - 1
        data = subprocess.run(['curl', '-sL', '-r', f'{s.pos}-{end}', s.url], capture_output=True).stdout
        b[:len(data)] = data; s.pos += len(data); return len(data)
def open_zip(url):
    return zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=1 << 22))
