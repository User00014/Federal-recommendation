import argparse
import csv
import math


def build_args():
    p = argparse.ArgumentParser(description="根据已完成实验估算全批次时间和费用")
    p.add_argument("--summary", required=True, help="batch_summary.csv 路径")
    p.add_argument("--planned-runs", type=int, required=True, help="计划总实验次数")
    p.add_argument("--price-per-hour", type=float, required=True, help="实例单价（元/小时）")
    return p.parse_args()


def main():
    args = build_args()

    done_durations = []
    failed = 0

    with open(args.summary, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            if status == "done":
                try:
                    done_durations.append(float(row.get("duration_sec") or 0.0))
                except Exception:
                    pass
            elif status == "failed":
                failed += 1

    if not done_durations:
        print("[ERR] summary 中没有可用的已完成实验时长，无法估算。")
        return

    avg_sec = sum(done_durations) / len(done_durations)
    total_sec = avg_sec * args.planned_runs
    total_hour = total_sec / 3600.0
    total_cost = total_hour * args.price_per_hour

    print(f"[INFO] 已完成样本数: {len(done_durations)}")
    print(f"[INFO] 失败样本数: {failed}")
    print(f"[INFO] 单组平均耗时: {avg_sec / 60.0:.2f} 分钟")
    print(f"[EST ] 预计总时长: {total_hour:.2f} 小时")
    print(f"[EST ] 预计总费用: {total_cost:.2f} 元")


if __name__ == "__main__":
    main()
