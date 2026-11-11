# MIT License, Copyright (c) 2026 Stefan Teleman (full text: gnfs/sieve/sieve_core.cu)
# relcheck.py <factor base file> <savefile>: checks every NFS relation (plain or gzip).
# check msieve relations: each listed prime divides its side's norm,
# and the rest of each norm has only prime factors < 256
import sys, gzip
def readfb(path):
    R, A = {}, {}
    for l in open(path):
        t = l.split()
        if len(t) == 2 and t[0][0] in 'RA' and t[0][1:].isdigit():
            (R if t[0][0] == 'R' else A)[int(t[0][1:])] = int(t[1])
    return [R[k] for k in sorted(R)], [A[k] for k in sorted(A)]
def norm(c, a, b):
    d = len(c) - 1
    return abs(sum(c[k] * a**k * b**(d - k) for k in range(d + 1)))
small = [p for p in range(2, 256) if all(p % q for q in range(2, p))]
rat, alg = readfb(sys.argv[1])
ok = bad = 0
opener = gzip.open if open(sys.argv[2], 'rb').read(2) == b'\x1f\x8b' else open
for line in opener(sys.argv[2], 'rt'):
    line = line.strip()
    if not line or line[0] in 'N#' or line.count(':') < 2:  # free relations (b = 0) from filtering
        continue
    ab, r, s = line.split(':')
    a, b = map(int, ab.split(','))
    good = True
    for c, fs in ((rat, r), (alg, s)):
        n = norm(c, a, b)
        seen = set()
        for f in fs.split(','):
            if not f:
                continue
            p = int(f, 16)
            if n % p and p not in seen:
                good = False
            seen.add(p)
            while n % p == 0:
                n //= p
        for p in small:
            while n % p == 0:
                n //= p
        if n != 1:
            good = False
    if good:
        ok += 1
    else:
        bad += 1
        if bad <= 3:
            print("bad:", line)
print(f"{ok} valid, {bad} invalid")
