# -*- coding: utf-8 -*-
"""
aggregate_seeds.py -- pool seeds that were run separately into one mean +/- std summary.

Because run_trajectory.py writes each seed to its own dir (EXP/seed{N}/arm_{arm}/arm_results.json),
you can run seeds one at a time (safer on a 16GB box) and combine them afterwards here -- no
recomputation. Reads every seed present under --exp_dir and prints the same summary + plot that
the all-at-once run would, over whatever seeds it finds.

Usage:
  # after running --seeds 0, then --seeds 1, then --seeds 2 into the same --exp_dir traj_avg
  python aggregate_seeds.py --exp_dir traj_avg --rounds 3
"""
import argparse, glob, json, os, re
import statistics as st


def load(exp_dir):
    """exp_dir/seed{N}/arm_{arm}/arm_results.json -> {arm: {seed: rows}}"""
    arms = {}
    for path in glob.glob(os.path.join(exp_dir, "seed*", "arm_*", "arm_results.json")):
        m = re.search(r"seed(\d+)[/\\]arm_([^/\\]+)[/\\]arm_results\.json$", path)
        if not m:
            continue
        seed, arm = int(m.group(1)), m.group(2)
        try:
            rows = json.load(open(path))["rows"]
        except Exception as e:
            print(f"[skip] {path}: {e}"); continue
        arms.setdefault(arm, {})[seed] = {"rows": rows}
    return arms


def summarize(per_seed, rounds):
    out = []
    for r in range(rounds):
        sa = [s["rows"][r]["student_acc"] for s in per_seed.values()
              if len(s["rows"]) > r and s["rows"][r]["student_acc"] is not None]
        ta = [s["rows"][r]["teacher_acc"] for s in per_seed.values()
              if len(s["rows"]) > r and s["rows"][r]["teacher_acc"] is not None]
        if not sa:
            continue
        out.append({"round": r, "n": len(sa),
                    "student_mean": st.mean(sa),
                    "student_std": (st.pstdev(sa) if len(sa) > 1 else 0.0),
                    "student_sem": (st.pstdev(sa)/len(sa)**0.5 if len(sa) > 1 else 0.0),
                    "teacher_mean": (st.mean(ta) if ta else None),
                    "student_all": sorted(sa)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()

    arms = load(a.exp_dir)
    if not arms:
        raise SystemExit(f"no seed results found under {a.exp_dir}/seed*/arm_*/arm_results.json")
    summary = {arm: summarize(ps, a.rounds) for arm, ps in arms.items()}

    print(f"\n=== pooled over seeds found in {a.exp_dir} ===")
    for arm, ps in arms.items():
        print(f"  {arm}: seeds {sorted(ps)}")
    print("\n--- student accuracy (mean +/- std [+/- sem], over seeds) ---")
    for arm, rows in summary.items():
        print(f"\n[{arm}]")
        for x in rows:
            vals = " ".join(f"{v:.3f}" for v in x["student_all"])
            print(f"  round {x['round']}: {x['student_mean']:.3f} +/- {x['student_std']:.3f} "
                  f"(sem {x['student_sem']:.3f}, n={x['n']}; teacher {x['teacher_mean']:.3f})  [{vals}]")

    armnames = list(summary)
    if len(armnames) == 2:
        a0, a1 = armnames
        print(f"\n  gap ({a1} - {a0}) per round  (negative = {a1} poisons harder):")
        s0 = {x["round"]: x for x in summary[a0]}
        s1 = {x["round"]: x for x in summary[a1]}
        for r in sorted(set(s0) & set(s1)):
            d = s1[r]["student_mean"] - s0[r]["student_mean"]
            pooled_sem = (s0[r]["student_sem"]**2 + s1[r]["student_sem"]**2) ** 0.5
            sig = "exceeds 1 SEM" if abs(d) > pooled_sem else "within noise"
            print(f"    round {r}: {d:+.3f}  (pooled sem {pooled_sem:.3f})  -> {sig}")
        print("\n  note: 'exceeds 1 SEM' is a weak bar; for a real claim want |gap| > ~2x pooled sem")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for arm, rows in summary.items():
            xs = [x["round"] for x in rows]
            ys = [x["student_mean"] for x in rows]
            es = [x["student_std"] for x in rows]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=arm)
        ax.set_xlabel("re-aim round"); ax.set_ylabel("distilled student accuracy")
        ax.set_title("Frozen vs re-aimed poison (mean +/- std over seeds; lower = stronger)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(a.exp_dir, "trajectory_pooled.png")
        fig.savefig(out, dpi=130)
        print(f"\nwrote {out}")
    except Exception as e:
        print(f"[plot] skipped ({e})")


if __name__ == "__main__":
    main()