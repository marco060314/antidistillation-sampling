# -*- coding: utf-8 -*-
"""
grad_cos.py -- why does the re-aimed (oracle) poison differ from the frozen one?

Turns the "signal collapse vs redundancy vs just-noise" debate into a measurement, using
the per-round gradients run_trajectory.py already saved on disk.

Computes, between two saved gradient dicts (e.g. oracle round0 = frozen-equivalent init grad,
and oracle round1/round2 = re-aimed grads):
  * global cosine  cos(g_a, g_b)
  * global norms and the norm ratio  ||g_b|| / ||g_a||
  * per-tensor cosine distribution (how many tensors stayed aligned vs went orthogonal)

Reading:
  cos ~ 1, norm ratio ~ 1   -> re-aiming barely changed g (no staleness to exploit; student
                               didn't move much)
  cos ~ 1, norm ratio << 1  -> SAME direction but shrunken: the useful "break good reasoning"
                               signal collapsed as the student was degraded. renorm then inflates
                               a smaller-signal vector. (the 'signal collapse' story)
  cos ~ 0                    -> re-aimed g drifted to an unrelated direction -> renormed poison is
                               largely a random-ish perturbation (~ random floor = does little)
  cos < 0                   -> re-aimed g points OPPOSITE the init poison -> re-aiming partially
                               UNDOES the frozen attack's direction

Pairwise across all three grads tells the trajectory: init -> r1 -> r2.

Usage:
  python grad_cos.py traj_avg/seed0/arm_oracle/round0/grads.pt \
                     traj_avg/seed0/arm_oracle/round1/grads.pt \
                     traj_avg/seed0/arm_oracle/round2/grads.pt
  # or compare frozen vs oracle at the same round:
  python grad_cos.py traj_avg/seed0/arm_frozen/round0/grads.pt \
                     traj_avg/seed0/arm_oracle/round1/grads.pt
"""
import argparse
import torch


def load(path):
    return torch.load(path, map_location="cpu")


def gnorm(g):
    return float(sum(float(torch.sum(v.float() ** 2)) for v in g.values()) ** 0.5)


def cosine(ga, gb):
    keys = [k for k in ga if k in gb and ga[k].shape == gb[k].shape]
    dot = sum(float(torch.sum(ga[k].float() * gb[k].float())) for k in keys)
    na = sum(float(torch.sum(ga[k].float() ** 2)) for k in keys) ** 0.5
    nb = sum(float(torch.sum(gb[k].float() ** 2)) for k in keys) ** 0.5
    return dot / (na * nb + 1e-20), keys


def per_tensor_cos(ga, gb, keys):
    out = []
    for k in keys:
        a, b = ga[k].float().flatten(), gb[k].float().flatten()
        d = float(a @ b); na = float(a.norm()); nb = float(b.norm())
        if na > 1e-12 and nb > 1e-12:
            out.append((k, d / (na * nb)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grads", nargs="+", help="2+ saved grad .pt files, in trajectory order")
    a = ap.parse_args()
    if len(a.grads) < 2:
        raise SystemExit("give at least 2 gradient files")

    gs = [(p, load(p)) for p in a.grads]
    print("norms:")
    for p, g in gs:
        print(f"  {gnorm(g):10.4g}   {p}")

    def label(p):
        parts = p.replace("\\", "/").split("/")
        return parts[-2] if len(parts) >= 2 else parts[-1]

    print("\npairwise cosine (consecutive + vs-first):")
    pairs = [(0, i) for i in range(1, len(gs))] + [(i, i + 1) for i in range(1, len(gs) - 1)]
    seen = set()
    for i, j in pairs:
        if (i, j) in seen: continue
        seen.add((i, j))
        c, keys = cosine(gs[i][1], gs[j][1])
        nr = gnorm(gs[j][1]) / (gnorm(gs[i][1]) + 1e-20)
        ptc = per_tensor_cos(gs[i][1], gs[j][1], keys)
        vals = sorted(x[1] for x in ptc)
        frac_aligned = sum(1 for v in vals if v > 0.5) / max(len(vals), 1)
        frac_orth = sum(1 for v in vals if abs(v) < 0.2) / max(len(vals), 1)
        med = vals[len(vals)//2] if vals else float("nan")
        print(f"\n  g[{i}] vs g[{j}]   ({label(a.grads[i])} -> {label(a.grads[j])})")
        print(f"    global cosine     = {c:+.4f}")
        print(f"    norm ratio b/a    = {nr:.3f}")
        print(f"    per-tensor cos: median {med:+.3f}, "
              f"{frac_aligned:.0%} aligned(>0.5), {frac_orth:.0%} ~orthogonal(|c|<0.2)")
        # verdict
        if c > 0.8 and nr > 0.7:
            v = "barely changed -> little staleness; student didn't move much"
        elif c > 0.6 and nr < 0.5:
            v = "SAME direction, shrunken norm -> signal-collapse story (useful poison signal decayed)"
        elif abs(c) < 0.2:
            v = "near-orthogonal -> re-aimed g unrelated to init; renormed poison ~ random-ish"
        elif c < -0.1:
            v = "OPPOSITE -> re-aiming partially reverses the frozen attack direction"
        else:
            v = "intermediate -> partial drift; inspect per-tensor spread"
        print(f"    -> {v}")


if __name__ == "__main__":
    main()