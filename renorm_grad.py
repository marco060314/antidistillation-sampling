# -*- coding: utf-8 -*-
"""
renorm_grad.py -- rescale a saved ADS gradient to a fixed global norm.

Why this exists: in the adaptive arms (B fresh-proxy, D oracle) we RECOMPUTE the
gradient at the model's current weights each round via save_grad.py. But ||grad||
drifts along the trajectory (it shrinks as the model converges), and gentraces.py's
finite-difference term  (lam/2eps)*(logp(theta+eps*g) - logp(theta-eps*g))  assumes
eps*g lands in the valid Taylor window (Fig. 7 of the paper). If ||g|| changes, the
effective perturbation size changes and eps is no longer calibrated.

Fix: after each recompute, rescale g to the SAME global norm as the round-0 gradient
(pass --ref_grad round0_grads.pt) or to an explicit --target_norm. This keeps eps
fixed across rounds and arms, so any difference in poisoning is due to gradient
DIRECTION (re-aiming), not perturbation magnitude.

The poison strength itself is still controlled by lam downstream; this only fixes ||g||.

Usage:
  # match the round-0 gradient's norm (recommended)
  python renorm_grad.py recomputed_grads.pt --ref_grad round0_grads.pt --out grads_renorm.pt
  # or set an explicit target global L2 norm
  python renorm_grad.py recomputed_grads.pt --target_norm 12.6 --out grads_renorm.pt
  # just report the norm, change nothing
  python renorm_grad.py grads.pt --report
"""
import argparse
import torch


def global_norm(grads):
    return float(sum(float(torch.sum(g.float() ** 2)) for g in grads.values()) ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grads", help="path to a saved grad dict (.pt) from save_grad.py")
    ap.add_argument("--ref_grad", default=None, help="rescale to this grad's global norm")
    ap.add_argument("--target_norm", type=float, default=None, help="explicit target global L2 norm")
    ap.add_argument("--out", default=None, help="output path (default: overwrite input)")
    ap.add_argument("--report", action="store_true", help="print norm and exit, no write")
    args = ap.parse_args()

    grads = torch.load(args.grads, map_location="cpu")
    cur = global_norm(grads)
    print(f"[renorm] {args.grads}: global norm = {cur:.6e}  ({len(grads)} tensors)")
    if args.report:
        return

    if args.ref_grad is not None:
        ref = torch.load(args.ref_grad, map_location="cpu")
        target = global_norm(ref)
        print(f"[renorm] ref {args.ref_grad}: global norm = {target:.6e}")
    elif args.target_norm is not None:
        target = args.target_norm
    else:
        raise SystemExit("provide --ref_grad or --target_norm (or --report)")

    if cur < 1e-20:
        raise SystemExit("[renorm] current grad norm ~0; refusing to rescale")
    scale = target / cur
    out = {k: (v.float() * scale).to(v.dtype) for k, v in grads.items()}
    assert abs(global_norm(out) - target) / max(target, 1e-12) < 2e-2, "rescale failed"
    dest = args.out or args.grads
    torch.save(out, dest)
    print(f"[renorm] scaled by {scale:.6e} -> global norm = {global_norm(out):.6e}; wrote {dest}")


if __name__ == "__main__":
    main()