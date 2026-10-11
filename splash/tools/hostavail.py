#!/usr/bin/env python3
"""Splash's own startup estimate of reclaimable host memory (runtime/engine/MemoryGovernor.cpp):
free + file-backed + purgeable + compression savings. Splash 1.2.x needs ~16 GiB of it to start."""
import re, subprocess, sys
d = {m.group(1): int(m.group(2)) for m in re.finditer(r"(.+?):\s+(\d+)\.", subprocess.check_output(["vm_stat"], text=True))}
free = d["Pages free"] - d["Pages speculative"]; fb = d["File-backed pages"]; pu = d["Pages purgeable"]
an = d["Anonymous pages"]; cp = d["Pages occupied by compressor"]; cd = d["Pages stored in compressor"]
sav = an // 2 if (not cp or cd // 2 >= cp) else (0 if cd <= cp else an - an * cp // cd)
g = lambda x: x * 16384 / 2**30
if "--gib" in sys.argv:            # integer GiB, for scripts
    print(int(g(free + fb + pu + sav))); sys.exit()
print(f"free {g(free):.1f} filebacked {g(fb):.1f} purgeable {g(pu):.1f} compress-savings {g(sav):.1f} => estimate {g(free+fb+pu+sav):.1f} GiB;"
      f" compressor {g(cp):.1f} GiB holding {g(cd):.1f}; wired {g(d['Pages wired down']):.1f}; anonymous {g(an):.1f}")
