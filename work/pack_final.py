"""Assemble the frozen submission package (code + models) into work/final_submission/ and write an
MD5 manifest. LightGBM boosters are gzipped (each < 100 MB, GitHub's hard limit); predict_final.py
transparently reads .txt.gz. Usage: python pack_final.py
"""
import gzip, hashlib, shutil, json
from pathlib import Path
WORK = Path(__file__).resolve().parent
ART = WORK / "final_artifacts"
OUT = WORK / "final_submission"
CODE = ["final_features.py", "final_blend.py", "train_final.py", "predict_final.py", "rehearse.py",
        "common.py", "build_reference_ids.py", "FINAL_PACKAGE_README.md", "pack_final.py"]
SMALL = ["prep.json", "genes.json", "holdout_A_ids.json", "trained_with_test_ids.json",
         "feature_names_static.npy", "prediction_train_time.csv"]

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "final_artifacts").mkdir(parents=True)
manifest = {}
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
for f in CODE:
    shutil.copy(WORK / f, OUT / f); manifest[f] = md5(OUT / f)
for f in SMALL:
    shutil.copy(ART / f, OUT / "final_artifacts" / f); manifest["final_artifacts/" + f] = md5(OUT / "final_artifacts" / f)
for p in sorted(ART.glob("*.txt")):                       # LightGBM boosters -> gzip
    dst = OUT / "final_artifacts" / (p.name + ".gz")
    with open(p, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo)
    manifest["final_artifacts/" + p.name + " (uncompressed)"] = md5(p)
    if dst.stat().st_size > 95 * 1048576:                    # GitHub hard limit is 100 MB -> split
        data = dst.read_bytes(); dst.unlink(); chunk = 90 * 1048576
        for i in range(0, len(data), chunk):
            part = dst.with_name(dst.name + f".part{i // chunk:02d}"); part.write_bytes(data[i:i + chunk])
            manifest["final_artifacts/" + part.name] = md5(part)
    else:
        manifest["final_artifacts/" + dst.name] = md5(dst)
for p in sorted(ART.glob("*.pt")) + sorted(ART.glob("*.features.json")) + sorted(ART.glob("*.meta.json")):
    shutil.copy(p, OUT / "final_artifacts" / p.name); manifest["final_artifacts/" + p.name] = md5(p)
json.dump(manifest, open(OUT / "MANIFEST.md5.json", "w"), indent=1)
total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1048576
big = [(p.name, p.stat().st_size / 1048576) for p in OUT.rglob("*") if p.is_file() and p.stat().st_size > 50 * 1048576]
print(f"package: {OUT}  total={total:.0f} MB  files={len(manifest)}  >50MB: {[(n, round(s)) for n, s in big]}")
