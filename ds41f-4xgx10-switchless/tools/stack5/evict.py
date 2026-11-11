#!/usr/bin/env python3
"""evict.py DIR... : drop the page cache of every regular file under the given dirs (posix_fadvise DONTNEED, no sudo).
CUDA on GB10 allocates only from free pages, and the engine sizes its session store from free device memory at start,
so big files read by a bench or a trainer on a rank must be evicted before the engine starts there."""
import os, sys
os.sync()
n = b = 0
for d in sys.argv[1:]:
    d = os.path.expanduser(d)
    for root, _, files in os.walk(d) if os.path.isdir(d) else [(os.path.dirname(d), [], [os.path.basename(d)])]:
        for f in files:
            p = os.path.join(root, f)
            try:
                if os.path.islink(p) or not os.path.isfile(p): continue
                sz = os.path.getsize(p)
                if sz < 1 << 20: continue
                fd = os.open(p, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1; b += sz
            except OSError: pass
print("evicted", n, "files,", round(b / 2**30, 1), "GiB")
