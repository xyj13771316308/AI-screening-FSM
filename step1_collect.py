# step1_collect.py
import os
import time
import pandas as pd
import socket
import urllib3.util.connection as urllib3_cn

def _force_ipv4():
    return socket.AF_INET

urllib3_cn.allowed_gai_family = _force_ipv4
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

# ===== 配置区 =====
API_KEY = os.environ.get("MP_API_KEY")
if not API_KEY:
    raise RuntimeError("没找到 MP_API_KEY，请先: export MP_API_KEY=你的key")

METALS = ["Zn", "Cu", "Zr", "Fe", "Co", "Ni", "Cr", "Al",
          "Mg", "Cd", "Mn", "Ti", "V", "In", "Mo", "W"]

LIMIT = 10
MAX_RETRY = 5
OUT = "data/mofs.parquet"
# ===================

os.makedirs("data", exist_ok=True)


def search_with_retry(mpr, metal, limit):
    for attempt in range(MAX_RETRY):
        try:
            return mpr.materials.summary.search(
                elements=[metal, "C", "O"],
                fields=["material_id", "formula_pretty", "structure",
                        "density", "volume"],
                chunk_size=(limit or 1000),
            )
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"    [retry {attempt+1}/{MAX_RETRY}] {type(e).__name__}, {wait}s 后重试...")
            time.sleep(wait)
    print(f"    [skip] {metal} 多次重试仍失败，跳过")
    return []


seen = {}
skipped = 0

for attempt in range(MAX_RETRY):
    try:
        mpr = MPRester(API_KEY)
        break
    except Exception as e:
        wait = 10 * (attempt + 1)
        print(f"[retry {attempt+1}/{MAX_RETRY}] 连接初始化失败: {type(e).__name__}, {wait}s 重试...")
        time.sleep(wait)
else:
    raise SystemExit("无法连接 Materials Project API，请检查网络/代理后重试")

with mpr:
    for metal in METALS:
        print(f"[mp] querying: elements=[{metal}, C, O] ...")
        docs = search_with_retry(mpr, metal, LIMIT)
        new = 0
        for d in docs:
            if d.material_id in seen:
                continue
            try:
                cif_writer = CifWriter(d.structure)
                seen[d.material_id] = {
                    "id": d.material_id,
                    "formula": d.formula_pretty,
                    "metal": metal,
                    "density": d.density,
                    "volume": getattr(d, "volume", None),
                    "cif": cif_writer.__str__(),
                }
                new += 1
            except Exception as e:
                skipped += 1
                print(f"    [warn] {d.material_id} 结构转换失败: {e}")
        print(f"    -> new={new}")

df = pd.DataFrame(seen.values())
df.to_parquet(OUT)
print(f"[mp] loaded={len(df)} skipped={skipped} -> {OUT}")
print(df["metal"].value_counts())
