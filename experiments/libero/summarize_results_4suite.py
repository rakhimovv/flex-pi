"""
Summarize LIBERO 4-suite evaluation results.
Handles all suite names (not just the 5 standard LIBERO suites).

Usage:
    python experiments/libero/summarize_results_4suite.py --output_dir eval_results/libero_4suite
"""

import argparse
import collections
import json
import glob
import os


def summarize(output_dir):
    files = glob.glob(os.path.join(output_dir, "**", "*_results.json"), recursive=True)
    if not files:
        print(f"No result files found in {output_dir}")
        return

    suite_stats = collections.defaultdict(lambda: {"success": 0, "total": 0, "tasks": 0})
    task_rows = []

    for f in sorted(files):
        with open(f) as fp:
            d = json.load(fp)
        suite = d["task_suite"]
        suite_stats[suite]["success"] += d["successes"]
        suite_stats[suite]["total"] += d["total_episodes"]
        suite_stats[suite]["tasks"] += 1
        task_rows.append({
            "suite": suite,
            "task_id": d["task_id"],
            "description": d.get("task_description", ""),
            "success": d["successes"],
            "total": d["total_episodes"],
            "rate": 100.0 * d["successes"] / d["total_episodes"] if d["total_episodes"] > 0 else 0,
        })

    # Per-task table
    print("\n=== Per-Task Success Rates ===")
    print("%-35s %5s  %-45s %8s" % ("Suite", "Task", "Description", "Rate"))
    print("-" * 100)
    for row in sorted(task_rows, key=lambda r: (r["suite"], r["task_id"])):
        print("%-35s %5d  %-45s %7.1f%%" % (
            row["suite"], row["task_id"],
            (row["description"] or "")[:45], row["rate"]
        ))

    # Per-suite summary
    print("\n=== Per-Suite Summary ===")
    print("%-40s %5s %10s %8s %8s" % ("Suite", "Tasks", "Success", "Total", "Rate"))
    print("-" * 75)
    total_success = total_total = 0
    for suite, s in sorted(suite_stats.items()):
        rate = 100.0 * s["success"] / s["total"] if s["total"] > 0 else 0
        print("%-40s %5d %10d %8d %7.1f%%" % (
            suite, s["tasks"], s["success"], s["total"], rate
        ))
        total_success += s["success"]
        total_total += s["total"]

    overall = 100.0 * total_success / total_total if total_total > 0 else 0
    print("-" * 75)
    print("%-40s %5d %10d %8d %7.1f%%" % (
        "OVERALL", sum(s["tasks"] for s in suite_stats.values()),
        total_success, total_total, overall
    ))

    # Save CSV
    csv_path = os.path.join(output_dir, "summary_4suite.csv")
    with open(csv_path, "w") as f:
        f.write("suite,tasks_completed,success,total,success_rate\n")
        for suite, s in sorted(suite_stats.items()):
            rate = 100.0 * s["success"] / s["total"] if s["total"] > 0 else 0
            f.write(f"{suite},{s['tasks']},{s['success']},{s['total']},{rate:.2f}\n")
    print(f"\nCSV saved to: {csv_path}")

    # Save JSON
    json_path = os.path.join(output_dir, "summary_4suite.json")
    with open(json_path, "w") as f:
        json.dump({
            "output_dir": output_dir,
            "overall_success_rate": overall,
            "total_success": total_success,
            "total_trials": total_total,
            "suites": {
                suite: {
                    "tasks_completed": s["tasks"],
                    "success": s["success"],
                    "total": s["total"],
                    "success_rate": 100.0 * s["success"] / s["total"] if s["total"] > 0 else 0,
                }
                for suite, s in sorted(suite_stats.items())
            }
        }, f, indent=2)
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory containing evaluation results")
    args = parser.parse_args()
    summarize(args.output_dir)
