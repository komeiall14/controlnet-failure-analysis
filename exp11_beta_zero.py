"""torch.baddbmm が beta=0 のときに加算元を無視するかを直接調べる。

BLAS の規約では beta = 0 のとき加算元 C は参照されない。したがって
C の中身が何であっても結果は変わらないはずである。diffusers の
get_attention_scores が C を未初期化のまま渡しているのも、この規約に
基づく。規約が守られていれば中身は無害である。

ここでは同じ q・k に対し、加算元を zeros にした場合と NaN で埋めた場合の
結果が一致するかを、精度（fp16 / fp32）とデバイス（MPS / CPU）の
4 通りで比べる。一致しなければ、その実装は beta=0 を守っていない。

モデルを読まないので数秒で終わる。

出力: results/exp11_beta_zero.csv
    python3 exp11_beta_zero.py
"""
import csv
import os

import torch

import pipeline as P


def check(dtype, device, fill):
    torch.manual_seed(0)
    q = torch.randn(2, 8, 16, dtype=dtype, device=device)
    k = torch.randn(2, 8, 16, dtype=dtype, device=device)
    kt = k.transpose(-1, -2)
    base = torch.baddbmm(torch.zeros(2, 8, 8, dtype=dtype, device=device),
                         q, kt, beta=0, alpha=0.25)
    dirty = torch.baddbmm(torch.full((2, 8, 8), fill, dtype=dtype, device=device),
                          q, kt, beta=0, alpha=0.25)
    # NaN 同士は == で比較できないので、埋めてから比べる
    same = torch.equal(base.float().nan_to_num(-9e9), dirty.float().nan_to_num(-9e9))
    return dict(dtype=str(dtype).split(".")[-1], device=device, fill=str(fill),
                out_has_nan=bool(torch.isnan(dirty).any()),
                matches_zeros=same, honors_beta0=same)


def main():
    rows = []
    for dtype in (torch.float16, torch.float32):
        for device in ("mps", "cpu"):
            for fill in (float("nan"), float("inf"), float("-inf"), 1e4):
                rows.append(check(dtype, device, fill))
    for r in rows:
        mark = "規約どおり" if r["honors_beta0"] else "★規約違反"
        print(f"  {r['dtype']:7s} {r['device']:3s} 加算元={r['fill']:>5s}  "
              f"出力にNaN={str(r['out_has_nan']):5s}  zeros版と一致={str(r['matches_zeros']):5s}"
              f"  {mark}")
    path = os.path.join(P.OUT, "exp11_beta_zero.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path} ({len(rows)} rows)")

    mps_bad = [r for r in rows if r["device"] == "mps" and not r["honors_beta0"]]
    cpu_ok = [r for r in rows if r["device"] == "cpu" and r["honors_beta0"]]
    print(f"\n  MPS で規約が守られなかった条件 {len(mps_bad)}/"
          f"{len([r for r in rows if r['device'] == 'mps'])}")
    print(f"  CPU で規約が守られた条件     {len(cpu_ok)}/"
          f"{len([r for r in rows if r['device'] == 'cpu'])}")
    if mps_bad and len(cpu_ok) == len([r for r in rows if r["device"] == "cpu"]):
        print("  → beta=0 を守らないのは MPS の実装で、精度には依存しない。"
              "fp16 でだけ壊れて見えるのは、そのメモリに非有限値が残っているかの違いである")


if __name__ == "__main__":
    main()
