# -*- coding: utf-8 -*-
"""
trace_diff.py -- how different are two ADS poison trace sets?

The staleness question has a hidden upstream check: re-aiming changes the GRADIENT, but ADS
turns the gradient into POISONED TOKENS via the teacher's sampling. If frozen vs oracle
gradients produce nearly identical poisoned *text*, then "frozen ~= oracle students" is
explained at the source -- the token channel laundered away the directional info that
re-aiming provides -- and that's a finding about ADS itself, not about the student.

This compares two trace datasets (same problems, same order assumed) and reports:
  * fraction of problems where the two traces reach a DIFFERENT final boxed answer
  * fraction where one is correct and the other isn't (the bit that matters for distillation)
  * mean normalized edit distance between the trace texts (0 = identical, 1 = totally different)
  * mean token-length difference
  * a few example problems where they diverge

Usage:
  python trace_diff.py traj_full/seed0/arm_frozen/round1/poison_train \
                       traj_full/seed0/arm_oracle/round1/poison_train --colname trace
"""
import argparse, re
import datasets


def boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        return nums[-1].replace(",", "") if nums else None
    j, depth, buf = i + 7, 1, ""
    while j < len(text) and depth > 0:
        c = text[j]
        if c == "{": depth += 1
        elif c == "}": depth -= 1
        if depth > 0: buf += c
        j += 1
    m = re.search(r"-?\d[\d,]*\.?\d*", buf)
    return m.group(0).replace(",", "") if m else buf.strip()


def norm_edit(a, b):
    """normalized Levenshtein on whitespace tokens (cheap, good enough for 'how different')."""
    A, B = a.split(), b.split()
    if not A and not B: return 0.0
    n, m = len(A), len(B)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (A[i-1] != B[j-1]))
        prev = cur
    return prev[m] / max(n, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds_a"); ap.add_argument("ds_b")
    ap.add_argument("--colname", default="trace")
    ap.add_argument("--answer_colname", default="trace_af", help="answer-forced column if present")
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()

    A = datasets.load_from_disk(a.ds_a)
    B = datasets.load_from_disk(a.ds_b)
    n = min(len(A), len(B))
    print(f"comparing {n} traces\n  A = {a.ds_a}\n  B = {a.ds_b}")

    col = a.colname
    acol = a.answer_colname if a.answer_colname in A.column_names else a.colname

    diff_ans = 0; one_right = 0; eds = []; dlen = []; diverge = []
    for i in range(n):
        ta, tb = A[i][col], B[i][col]
        aa, ab = boxed(A[i][acol]), boxed(B[i][acol])
        if aa != ab:
            diff_ans += 1
            diverge.append((i, aa, ab))
        # correctness divergence if a gold solution is present
        if "solution" in A.column_names:
            gold = boxed(A[i]["solution"])
            if (aa == gold) != (ab == gold):
                one_right += 1
        eds.append(norm_edit(ta, tb))
        dlen.append(abs(len(ta.split()) - len(tb.split())))

    print(f"\n  different final answer:      {diff_ans}/{n} = {diff_ans/n:.1%}")
    if "solution" in A.column_names:
        print(f"  one correct, other not:      {one_right}/{n} = {one_right/n:.1%}  <- the bit distillation sees")
    print(f"  mean normalized edit dist:   {sum(eds)/n:.3f}   (0=identical text, 1=totally different)")
    print(f"  mean |token-length diff|:    {sum(dlen)/n:.1f} tokens")

    print("\n  interpretation:")
    ed = sum(eds)/n
    if ed < 0.15 and diff_ans/n < 0.1:
        print("  -> traces are NEARLY IDENTICAL. The gradient difference did not survive into the")
        print("     poisoned text -> 'frozen ~= oracle student' is explained by the token channel,")
        print("     not by the student. Re-aiming changes g but not the emitted poison.")
    elif diff_ans/n > 0.3:
        print("  -> traces DIFFER substantially (answers diverge often). The gradient difference DID")
        print("     reach the text, so if students still come out equal it's downstream (subspace/LoRA/")
        print("     student barely moving), not the token channel.")
    else:
        print("  -> intermediate. Some directional info survives; inspect examples below.")

    if diverge:
        print(f"\n  example divergent problems (idx, ansA, ansB):")
        for i, aa, ab in diverge[:a.examples]:
            print(f"    #{i}: A={aa}  B={ab}")


if __name__ == "__main__":
    main()