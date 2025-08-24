import argparse, shutil, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TPL_THEORY = ROOT / "days" / "_TEMPLATE_THEORY"
TPL_BUILD  = ROOT / "days" / "_TEMPLATE_BUILD"
DAYS = ROOT / "days"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["theory", "build"], required=True)
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--slug", type=str, required=True, help="kebab-case, e.g., transformer-architecture")
    args = ap.parse_args()

    target = DAYS / f"day-{args.day:02d}-{args.slug}"
    if target.exists():
        print("Target already exists:", target)
        sys.exit(1)

    tpl = TPL_THEORY if args.type == "theory" else TPL_BUILD
    shutil.copytree(tpl, target)
    print("Created", target)

if __name__ == "__main__":
    main()
