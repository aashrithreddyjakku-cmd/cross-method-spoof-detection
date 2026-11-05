"""Download a SMALL, balanced subset of ASVspoof 2019 LA (+ In-the-Wild) without downloading the full zips.

Both archives (LA.zip 7.6 GB on Edinburgh DataShare, release_in_the_wild.zip 8.2 GB on Hugging Face) are read
remotely with HTTP range requests: only the zip index, the protocol / meta files and the selected audio files are
transferred. The result has the normal raw layout, so every other script works unchanged:

  data/raw/asvspoof2019/LA/ASVspoof2019_LA_{train,dev,eval}/flac/*.flac
  data/raw/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/*.txt        <- ONLY the selected lines
  data/raw/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/full/*.txt   <- original full protocols (reference)
  data/raw/in_the_wild/release_in_the_wild/{meta.csv,*.wav}          <- ONLY the selected clips

Selection (seed 1): inside each official partition, bona fide utterances and the spoofs of EVERY attack are
sampled evenly across speakers, so all 107 speakers and all 19 attacks stay present and P0/P1 splits still work.

  python scripts/download_subset.py --size small --dry-run     # show what would be downloaded
  python scripts/download_subset.py --size small               # about 4.8 k files, about 450 MB
  python scripts/download_subset.py --size tiny --no-itw       # about 1.6 k files
Re-running resumes (existing files are skipped). Same seed + size -> the same subset on every machine.
"""
import argparse
import io
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from spoofdet.common import paths

LA_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
ITW_URL = "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
PARTS = {"train": "ASVspoof2019.LA.cm.train.trn.txt", "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
         "eval": "ASVspoof2019.LA.cm.eval.trl.txt"}
#           (bona fide per partition, spoof per attack per partition)   train, dev, eval   + ITW per class
SIZES = {"tiny":   dict(train=(300, 60),  dev=(150, 30), eval=(400, 50),  itw=300),
         "small":  dict(train=(600, 120), dev=(300, 60), eval=(900, 100), itw=300),
         "medium": dict(train=(1500, 300), dev=(600, 120), eval=(2000, 250), itw=1500)}


# ------------------------------------------------------------------ remote zip over HTTP range requests
class RemoteZip:
    def __init__(self, url, retries=5):
        self.url, self.retries = url, retries
        self._resolve()

    def _resolve(self):                       # follow redirects once (DataShare / HF CDN); refreshed on 403
        for k in range(self.retries):
            req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0", "User-Agent": "spoofdet"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    cr = r.headers.get("Content-Range")
                    if r.status == 206 and cr:
                        self.final, self.size = r.geturl(), int(cr.split("/")[-1])
                        return
            except urllib.error.URLError:
                pass
            time.sleep(2 ** k)
        raise IOError(f"{self.url} does not answer range requests right now; try again later")

    def get(self, start, end):                # inclusive byte range
        for k in range(self.retries):
            try:
                req = urllib.request.Request(self.final, headers={"Range": f"bytes={start}-{end}", "User-Agent": "spoofdet"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise IOError(f"server ignored the range request (HTTP {r.status})")
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    self._resolve()           # signed CDN link expired
                if k == self.retries - 1:
                    raise
            except Exception:
                if k == self.retries - 1:
                    raise
            time.sleep(2 ** k)

    def infolist(self):
        """Read only the end of the zip (EOCD + central directory) and let zipfile parse it."""
        tail = self.get(max(0, self.size - (1 << 16)), self.size - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
        loc = tail.rfind(b"PK\x06\x07")       # zip64 locator -> zip64 end record holds the real values
        if loc != -1:
            z64_off = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
            z64 = self.get(z64_off, z64_off + 55)
            cd_size, cd_off = struct.unpack("<QQ", z64[40:56])
        start = min(cd_off, self.size - len(tail))
        blob = self.get(start, self.size - 1)
        return zipfile.ZipFile(_Window(blob, start, self.size)).infolist()

    def read(self, info: zipfile.ZipInfo) -> bytes:
        name_len = len(info.orig_filename.encode("utf-8" if info.flag_bits & 0x800 else "cp437"))
        head = 30 + name_len + 1024           # local extra field is usually short; re-fetch if not
        buf = self.get(info.header_offset, info.header_offset + head + info.compress_size)
        assert buf[:4] == b"PK\x03\x04", f"bad local header for {info.filename}"
        n, m = struct.unpack("<HH", buf[26:30])
        s = 30 + n + m
        if s + info.compress_size > len(buf):
            buf = self.get(info.header_offset, info.header_offset + s + info.compress_size)
        comp = buf[s:s + info.compress_size]
        if info.compress_type == zipfile.ZIP_STORED:
            data = comp
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompressobj(-15).decompress(comp)
        else:
            raise NotImplementedError(f"compression {info.compress_type} for {info.filename}")
        if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
            raise IOError(f"CRC mismatch for {info.filename}")
        return data


class _Window(io.RawIOBase):
    """File-like view of the zip's tail only (offsets are those of the full remote file)."""
    def __init__(self, blob, start, size):
        self.blob, self.start, self.size, self.pos = blob, start, size, start
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def read(self, n=-1):
        i = self.pos - self.start
        if i < 0:
            raise IOError("read before the downloaded window")
        out = self.blob[i:] if n is None or n < 0 else self.blob[i:i + n]
        self.pos += len(out)
        return out
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


# ------------------------------------------------------------------ selection
def spread_sample(df, n, by, seed):
    """n rows, spread as evenly as possible over the values of `by` (round-robin after a seeded shuffle)."""
    if len(df) <= n:
        return df
    d = df.sample(frac=1, random_state=seed)
    d = d.assign(_r=d.groupby(by).cumcount()).sort_values(["_r"], kind="stable")
    return d.head(n).drop(columns="_r")


def fetch_all(rz, jobs, workers, desc):
    """jobs: list of (ZipInfo, dest Path). Skips files that already exist with the right size."""
    todo = [(i, p) for i, p in jobs if not (p.exists() and p.stat().st_size == i.file_size)]
    print(f"{desc}: {len(jobs) - len(todo)} already present, downloading {len(todo)} "
          f"({sum(i.compress_size for i, _ in todo) / 1e6:.0f} MB)")

    def one(job):
        info, dest = job
        data = rz.read(info)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part"); tmp.write_bytes(data); tmp.replace(dest)

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, j) for j in todo]
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
            f.result()


def do_la(a, size, raw):
    rz = RemoteZip(a.la_url)
    print(f"LA.zip: {rz.size / 1e9:.2f} GB remote, reading index ...")
    infos = {i.filename: i for i in rz.infolist()}
    proto_dir = raw / "asvspoof2019" / "LA" / "ASVspoof2019_LA_cm_protocols"
    jobs, summary = [], []
    for part, proto in PARTS.items():
        key = next(k for k in infos if k.endswith("ASVspoof2019_LA_cm_protocols/" + proto))
        text = rz.read(infos[key]).decode()
        (proto_dir / "full").mkdir(parents=True, exist_ok=True)
        (proto_dir / "full" / proto).write_text(text)
        rows = pd.DataFrame([l.split()[:5] for l in text.splitlines() if l.strip()],
                            columns=["spk", "utt", "x", "attack", "key"])
        rows["line"] = [l for l in text.splitlines() if l.strip()]
        n_bona, n_per_attack = size[part]
        bona = spread_sample(rows[rows.key == "bonafide"], n_bona, "spk", a.seed)
        spoof = [spread_sample(g, n_per_attack, "spk", a.seed) for _, g in rows[rows.key == "spoof"].groupby("attack")]
        sel = pd.concat([bona] + spoof).sort_index()
        (proto_dir / proto).write_text("\n".join(sel.line) + "\n")
        audio_dir = f"ASVspoof2019_LA_{part}/flac"
        for u in sel.utt:
            k = next((c for c in (f"LA/{audio_dir}/{u}.flac", f"{audio_dir}/{u}.flac") if c in infos), None)
            if k is None:
                raise KeyError(f"{u}.flac not found in LA.zip")
            jobs.append((infos[k], raw / "asvspoof2019" / "LA" / audio_dir / f"{u}.flac"))
        summary.append(f"{part}: {len(bona)} bona fide + {len(sel) - len(bona)} spoof "
                       f"({sel[sel.key == 'spoof'].attack.nunique()} attacks x <= {n_per_attack}), "
                       f"{sel.spk.nunique()} speakers")
        print(summary[-1])
    total = sum(i.compress_size for i, _ in jobs)
    print(f"LA subset: {len(jobs)} files, {total / 1e6:.0f} MB")
    if a.dry_run:
        return
    fetch_all(rz, jobs, a.workers, "LA")
    note = (f"ASVspoof 2019 LA SUBSET (size={a.size}, seed={a.seed}) made by scripts/download_subset.py\n"
            f"source: {a.la_url}\n" + "\n".join(summary) + "\nfull protocols: ASVspoof2019_LA_cm_protocols/full/\n")
    (raw / "asvspoof2019" / "LA" / "SUBSET.txt").write_text(note)


def do_itw(a, n_per_class, raw):
    rz = RemoteZip(a.itw_url)
    print(f"release_in_the_wild.zip: {rz.size / 1e9:.2f} GB remote, reading index ...")
    infos = {i.filename: i for i in rz.infolist()}
    mkey = next(k for k in infos if k.endswith("meta.csv"))
    meta = pd.read_csv(io.BytesIO(rz.read(infos[mkey])))
    lcol = next(c for c in meta.columns if c.lower().strip() in ("label", "class", "key"))
    scol = next(c for c in meta.columns if "speaker" in c.lower())
    fcol = next(c for c in meta.columns if c.lower().strip() in ("file", "path", "filename"))
    sel = pd.concat([spread_sample(g, n_per_class, scol, a.seed) for _, g in meta.groupby(lcol)]).sort_index()
    prefix = mkey[: -len("meta.csv")]
    out = raw / "in_the_wild" / "release_in_the_wild"
    jobs = [(infos[prefix + f], out / f) for f in sel[fcol].astype(str)]
    print(f"ITW subset: {len(sel)} clips ({sel[lcol].value_counts().to_dict()}), {sel[scol].nunique()} speakers, "
          f"{sum(i.compress_size for i, _ in jobs) / 1e6:.0f} MB")
    if a.dry_run:
        return
    fetch_all(rz, jobs, a.workers, "ITW")
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / "meta.csv", index=False)
    meta.to_csv(out / "meta_full.csv.txt", index=False)       # not named meta.csv, so parse_itw ignores it
    (out / "SUBSET.txt").write_text(f"In-the-Wild SUBSET ({n_per_class} per class, seed={a.seed}) from {a.itw_url}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZES, default="small")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--raw", default=str(paths.RAW))
    ap.add_argument("--workers", type=int, default=16, help="parallel downloads")
    ap.add_argument("--no-itw", action="store_true")
    ap.add_argument("--no-la", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="only print what would be downloaded")
    ap.add_argument("--la-url", default=LA_URL)
    ap.add_argument("--itw-url", default=ITW_URL)
    a = ap.parse_args()
    raw = Path(a.raw); size = SIZES[a.size]
    if not a.no_la:
        do_la(a, size, raw)
    if not a.no_itw:
        do_itw(a, size["itw"], raw)
    print("done ->", raw)


if __name__ == "__main__":
    main()
