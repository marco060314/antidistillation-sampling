# -*- coding: utf-8 -*-
"""
run_trajectory.py -- end-to-end test of the staleness hypothesis at the ACCURACY level.

Wraps your existing scripts (save_grad.py -> gentraces.py -> distill.py -> gentraces.py)
in a round-by-round outer loop, so the poison can be re-aimed as the student distills.
This is the experiment that turns the per-step weight-space prediction (re-aiming helps
~1/cos ~ 10x in the first-order regime) into an actual student-accuracy curve.

ARMS (select with --arms). Each differs ONLY in which model the gradient is computed on
and used as the finite-difference base in gentraces:

  A frozen        : g = dL(proxy_init), computed ONCE, frozen.            base = proxy_init   (this is real ADS)
  B fresh_proxy   : g = dL(proxy_current), recomputed each round.          base = proxy_current
                    (proxy is distilled in parallel on the same poison as a surrogate student)
  C frozen_oracle : g = dL(student_init), computed ONCE, frozen.           base = student_init
  D oracle        : g = dL(student_current), recomputed each round.        base = student_current  (upper bound)

CRITICAL: gentraces' Delta-hat = (lam/2eps)*(logp(base+eps*g) - logp(base-eps*g)) only
estimates lam*<dL, dlogp(.;base)> when g was computed on `base`. So the gradient model and
gentraces.proxy_student MUST be the same checkpoint (handled here). A and B use the proxy;
C and D use the student. A/C freeze g at round 0; B/D recompute and RENORM each round
(renorm_grad.py, to round-0 norm) so eps stays calibrated and only DIRECTION changes.

Per round r (for each arm):
  1. gradient  -> save_grad.py on the arm's current base model; (B/D) renorm to round-0 norm
  2. poison    -> gentraces.py: teacher generates TRAIN traces, finite-diff base = arm base,
                  grad_path = step 1.  Records teacher accuracy under this poison.
  3. distill   -> distill.py: student CONTINUES from its previous-round weights, k epochs on
                  the new poison. (B also distills the proxy on the same poison.)
  4. evaluate  -> gentraces.py with lam=0, data_split=*_test, teacher = the distilled student,
                  -> student task accuracy (af_accuracy in the output .yaml).

Read-out: student accuracy vs round, per arm, all at (approximately) matched teacher accuracy.
  * D (and B) drive student accuracy BELOW A as rounds progress -> re-aiming compounds ->
    staleness matters end-to-end (supports the hypothesis).
  * all arms track A -> the frozen poison is as good as re-aimed -> freezing is fine.

MATCHED TEACHER ACCURACY: lam is fixed here and teacher accuracy is reported per round. To
compare strictly at matched teacher accuracy you must sweep lam per arm to a target (very
expensive: each lam regenerates traces). Start fixed-lam, then refine the crossover region.

COST WARNING: every round runs the teacher generator twice (poison + eval) plus a distill.
This is many GPU-hours per arm. Use --dry_run first to see the full plan; start with
--rounds 3 --max_samples small.

This orchestrator was validated at the command-construction / parsing level only (the author
could not run the 7B pipeline). Verify the hydra field names against your gen_config.yaml /
train_config.yaml before a real run; they are surfaced as constants below.
"""
import argparse, json, os, subprocess, sys, tempfile, shutil
import yaml


# ---- hydra/argparse field names used by your scripts (verify against your configs) ----
GENTRACES = "gentraces.py"     # hydra: teacher, proxy_student, grad_path, lam, eps, tau,
                               #        data_split, trace_path, trace_name, trace_colname,
                               #        answer_force, is_teacher, tokenizer, batch_size,
                               #        max_length, max_samples, seed, use_wandb
DISTILL   = "distill.py"       # hydra: student, tokenizer, train_traces, holdout_traces, lora,
                               #        lora_r, lora_alpha, lr, num_epochs, batch_size,
                               #        per_device_batch_size, model_path, model_name, exp_dir,
                               #        do_eval, seed, wandb
SAVE_GRAD = "save_grad.py"     # argparse: <holdout_config.yaml> --proxy_student --tokenizer
                               #           --trace_colname --batch_size ; writes <exp_dir>/student_grads.pt


def sh(cmd, dry, log):
    line = " ".join(str(c) for c in cmd)
    log.append(line)
    print(("[dry] " if dry else "[run] ") + line, flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def af_accuracy(trace_path, dry):
    """read stats.af_accuracy from <trace_path>.yaml written by gentraces.py."""
    if dry:
        return None
    with open(trace_path + ".yaml") as f:
        c = yaml.safe_load(f)
    return float(c["stats"]["af_accuracy"])


def write_holdout_cfg(base_cfg, exp_dir, holdout_traces, trace_colname, dst):
    """save_grad.py reads exp_dir + trace_path from its positional config; build one per call."""
    cfg = {}
    if base_cfg and os.path.exists(base_cfg):
        with open(base_cfg) as f:
            cfg = yaml.safe_load(f) or {}
    cfg.update({"exp_dir": exp_dir, "trace_path": holdout_traces, "trace_colname": trace_colname})
    with open(dst, "w") as f:
        yaml.safe_dump(cfg, f)
    return dst


# ------------------------------------------------------------------ command builders
def cmd_launch(a, script, overrides):
    return a.launcher.split() + [script] + overrides

def cmd_savegrad(a, holdout_cfg, base_model):
    return a.launcher.split() + [SAVE_GRAD, holdout_cfg,
            "--proxy_student", base_model, "--tokenizer", a.tokenizer,
            "--trace_colname", a.trace_colname, "--batch_size", str(a.grad_batch_size)]

def cmd_gentraces(a, *, teacher, base_model, grad_path, lam, data_split, trace_path,
                  trace_name, is_teacher, tau=None):
    tau = a.tau if tau is None else tau
    # poison-gen uses the teacher/proxy (Qwen) tokenizer; eval-gen uses the distilled
    # student's tokenizer (may be a different family).
    tok = a.tokenizer if is_teacher else a.student_tokenizer
    ov = [f"teacher={teacher}", f"tau={tau}", f"lam={lam}", f"eps={a.eps}",
          f"data_split={data_split}", f"trace_path={trace_path}", f"trace_name={trace_name}",
          f"trace_colname={a.trace_colname}", f"tokenizer={tok}", f"exp_dir={a.exp_dir}",
          f"answer_force={str(a.answer_force).lower()}", f"is_teacher={str(is_teacher).lower()}",
          f"batch_size={a.gen_batch_size}", f"max_length={a.max_length}",
          f"seed={a.seed}", "use_wandb=false"]
    if a.max_samples: ov.append(f"max_samples={a.max_samples}")
    if lam != 0:      ov += [f"proxy_student={base_model}", f"grad_path={grad_path}"]
    return cmd_launch(a, GENTRACES, ov)

def cmd_distill(a, *, student, train_traces, model_path, model_name):
    ov = [f"student={student}", f"tokenizer={a.student_tokenizer}", f"train_traces={train_traces}",
          f"holdout_traces={a.holdout_traces}", f"lora={str(a.lora).lower()}",
          f"lora_r={a.lora_r}", f"lora_alpha={a.lora_alpha}", f"lr={a.lr}",
          f"num_epochs={a.epochs_per_round}", f"batch_size={a.distill_batch_size}",
          f"per_device_batch_size={a.per_device_batch_size}", f"model_path={model_path}",
          f"model_name={model_name}", f"exp_dir={a.exp_dir}", f"max_length={a.max_length}",
          "do_eval=false", f"seed={a.seed}", "wandb=false"]
    return cmd_launch(a, DISTILL, ov)


# ------------------------------------------------------------------ one arm
ARM_BASE = {  # which model is the finite-diff base / gradient model, and whether g is frozen
    "frozen":        ("proxy",   True),
    "fresh_proxy":   ("proxy",   False),
    "frozen_oracle": ("student", True),
    "oracle":        ("student", False),
}

def run_arm(a, arm, seed, dry):
    base_kind, frozen = ARM_BASE[arm]
    a.seed = seed                                          # threaded into cmd_* (savegrad/gentraces/distill)
    work = os.path.join(a.exp_dir, f"seed{seed}", f"arm_{arm}")
    os.makedirs(work, exist_ok=True) if not dry else None
    log = []
    rows = []
    student_dir = a.student_init          # student continues from here, updated each round
    proxy_dir   = a.proxy_init            # only B advances this
    round0_grad = None
    frozen_grad = None

    for r in range(a.rounds):
        rd = os.path.join(work, f"round{r}")
        if not dry: os.makedirs(rd, exist_ok=True)
        base_model = (proxy_dir if base_kind == "proxy" else student_dir)
        if base_kind == "proxy" and frozen:   base_model = a.proxy_init
        if base_kind == "student" and frozen: base_model = a.student_init

        # 1) gradient
        grad_path = os.path.join(rd, "grads.pt")
        if frozen and frozen_grad is not None:
            grad_path = frozen_grad                       # reuse the round-0 frozen grad
        else:
            hc = os.path.join(rd, "holdout_cfg.yaml")
            if not dry:
                write_holdout_cfg(a.holdout_config, rd, a.holdout_traces, a.trace_colname, hc)
            else:
                print(f"[dry] write holdout cfg -> {hc} (exp_dir={rd})", flush=True)
            sh(cmd_savegrad(a, hc, base_model), dry, log)
            produced = os.path.join(rd, "student_grads.pt")   # save_grad writes <exp_dir>/student_grads.pt
            sh(["cp", produced, grad_path], dry, log)
            if r == 0:
                round0_grad = grad_path
                sh([sys.executable, "renorm_grad.py", grad_path, "--report"], dry, log)
            else:
                sh([sys.executable, "renorm_grad.py", grad_path, "--ref_grad", round0_grad,
                    "--out", grad_path], dry, log)
            if frozen:
                frozen_grad = grad_path

        # 2) poison TRAIN traces (teacher generates)
        poison = os.path.join(rd, "poison_train")
        sh(cmd_gentraces(a, teacher=a.teacher, base_model=base_model, grad_path=grad_path,
                         lam=a.lam, data_split=a.train_split, trace_path=poison,
                         trace_name=f"{arm}_r{r}_poison", is_teacher=True), dry, log)
        t_acc = af_accuracy(poison, dry)

        # 3) distill student (continue from previous weights), and (B) the proxy too
        smodel = os.path.join(rd, "student")
        sh(cmd_distill(a, student=student_dir, train_traces=poison,
                       model_path=smodel, model_name=f"{arm}_r{r}_student"), dry, log)
        student_dir = os.path.join(smodel, "final")
        if arm == "fresh_proxy":
            pmodel = os.path.join(rd, "proxy")
            sh(cmd_distill(a, student=proxy_dir, train_traces=poison,
                           model_path=pmodel, model_name=f"{arm}_r{r}_proxy"), dry, log)
            proxy_dir = os.path.join(pmodel, "final")

        # 4) evaluate student task accuracy (clean gen on test)
        ev = os.path.join(rd, "student_eval")
        sh(cmd_gentraces(a, teacher=student_dir, base_model="", grad_path="", lam=0,
                         data_split=a.test_split, trace_path=ev,
                         trace_name=f"{arm}_r{r}_eval", is_teacher=False, tau=a.eval_tau), dry, log)
        s_acc = af_accuracy(ev, dry)

        rows.append({"round": r, "teacher_acc": t_acc, "student_acc": s_acc,
                     "base_model": base_model, "grad": grad_path})
        print(f"[{arm}] round {r}: teacher_acc={t_acc} student_acc={s_acc}", flush=True)
        if not dry:
            with open(os.path.join(work, "arm_results.json"), "w") as f:
                json.dump({"arm": arm, "lam": a.lam, "eps": a.eps, "rows": rows}, f, indent=2)
    return {"arm": arm, "rows": rows, "commands": log}


def main():
    ap = argparse.ArgumentParser()
    # models  (0.5B defaults; orchestrator is size-agnostic -- override freely)
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--proxy_init", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--student_init", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--student_tokenizer", default=None,
                    help="tokenizer for the student (distill + eval); defaults to --tokenizer. "
                         "Set this when the student is a different family from the teacher/proxy.")
    # data / traces
    ap.add_argument("--holdout_traces", required=False, help="HF dataset dir of clean holdout traces (for dL and distill eval)")
    ap.add_argument("--holdout_config", default=None, help="base yaml for save_grad (merged with per-round exp_dir)")
    ap.add_argument("--train_split", default="gsm8k_train")
    ap.add_argument("--test_split",  default="gsm8k_test")
    ap.add_argument("--trace_colname", default="trace")
    ap.add_argument("--answer_force", action="store_true", default=True)
    # ADS knobs
    ap.add_argument("--lam", type=float, default=2.5e-3, help="fixed poison strength (sweep separately to match teacher acc)")
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--tau", type=float, default=0.6)
    # loop
    ap.add_argument("--arms", nargs="+", default=["frozen", "oracle"],
                    choices=list(ARM_BASE.keys()))
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--epochs_per_round", type=float, default=1.0, help="distill epochs between re-aims")
    # distill hyperparams
    ap.add_argument("--lora", action="store_true", default=True)
    ap.add_argument("--lora_r", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--distill_batch_size", type=int, default=16)
    ap.add_argument("--per_device_batch_size", type=int, default=2)
    # gen
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--grad_batch_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=200)
    # infra
    ap.add_argument("--exp_dir", default="trajectory_exp")
    ap.add_argument("--launcher", default="accelerate launch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="run the whole trajectory once per seed and report mean +/- std (e.g. 0 1 2 3 4)")
    ap.add_argument("--eval_tau", type=float, default=0.0,
                    help="temperature for the student eval generation; 0 = greedy/deterministic (recommended)")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else [a.seed]
    if a.student_tokenizer is None:
        a.student_tokenizer = a.tokenizer

    if not a.dry_run and not a.holdout_traces:
        ap.error("--holdout_traces is required for a real run")
    if not a.dry_run:
        os.makedirs(a.exp_dir, exist_ok=True)

    results = {}     # arm -> {seeds: {seed: rows}, summary: [per-round mean/std]}
    for arm in a.arms:
        per_seed = {}
        for sd in seeds:
            print(f"\n===== ARM: {arm} | seed {sd} =====", flush=True)
            per_seed[sd] = run_arm(a, arm, sd, a.dry_run)
        results[arm] = {"per_seed": per_seed, "summary": summarize(per_seed, a.rounds)}

    if not a.dry_run:
        out = {"config": {k: v for k, v in vars(a).items()}, "seeds": seeds, "results": results}
        with open(os.path.join(a.exp_dir, "trajectory_results.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.exp_dir}/trajectory_results.json", flush=True)
        print_summary(results)
        plot(results, a.exp_dir)
    else:
        total = sum(len(s["commands"]) for arm in results.values() for s in arm["per_seed"].values())
        print(f"\n[dry] planned {total} commands across {len(a.arms)} arm(s) x {len(seeds)} seed(s), "
              f"{a.rounds} rounds each. Remove --dry_run to execute.", flush=True)


def summarize(per_seed, rounds):
    """per-round mean/std of student_acc and teacher_acc across seeds."""
    import statistics as st
    out = []
    for r in range(rounds):
        sa = [s["rows"][r]["student_acc"] for s in per_seed.values()
              if len(s["rows"]) > r and s["rows"][r]["student_acc"] is not None]
        ta = [s["rows"][r]["teacher_acc"] for s in per_seed.values()
              if len(s["rows"]) > r and s["rows"][r]["teacher_acc"] is not None]
        if not sa:
            continue
        row = {"round": r, "n": len(sa),
               "student_mean": st.mean(sa), "student_std": (st.pstdev(sa) if len(sa) > 1 else 0.0),
               "teacher_mean": (st.mean(ta) if ta else None),
               "teacher_std": (st.pstdev(ta) if len(ta) > 1 else 0.0),
               "student_all": sa}
        out.append(row)
    return out


def print_summary(results):
    print("\n================ SUMMARY (student accuracy, mean +/- std over seeds) ================")
    for arm, res in results.items():
        print(f"\n[{arm}]")
        for row in res["summary"]:
            vals = " ".join(f"{v:.3f}" for v in row["student_all"])
            print(f"  round {row['round']}: {row['student_mean']:.3f} +/- {row['student_std']:.3f} "
                  f"(n={row['n']}; teacher {row['teacher_mean']:.3f})   [{vals}]")
    arms = list(results)
    if len(arms) == 2:
        print(f"\n  gap ({arms[1]} - {arms[0]}) per round (negative = {arms[1]} poisons harder):")
        s0 = {x["round"]: x for x in results[arms[0]]["summary"]}
        s1 = {x["round"]: x for x in results[arms[1]]["summary"]}
        for r in sorted(set(s0) & set(s1)):
            d = s1[r]["student_mean"] - s0[r]["student_mean"]
            pooled = (s0[r]["student_std"]**2 + s1[r]["student_std"]**2) ** 0.5
            sig = "  (within noise)" if abs(d) <= pooled else "  (exceeds pooled std)"
            print(f"    round {r}: {d:+.3f}  (pooled std {pooled:.3f}){sig}")


def plot(results, out):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped ({e})"); return
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for arm, res in results.items():
        s = res["summary"]
        if not s: continue
        xs = [r["round"] for r in s]
        ys = [r["student_mean"] for r in s]
        es = [r["student_std"] for r in s]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=arm)
    ax.set_xlabel("re-aim round"); ax.set_ylabel("distilled student accuracy")
    ax.set_title("Re-aiming vs frozen poison (mean +/- std over seeds; lower = stronger poison)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "trajectory.png"), dpi=130)
    print(f"wrote {out}/trajectory.png")


if __name__ == "__main__":
    main()