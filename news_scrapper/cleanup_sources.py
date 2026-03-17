r"""
cleanup_sources.py — removes article folders not in trusted_sources from data/raw

Usage:
  python cleanup_sources.py --data-dir data/raw --dry-run
  python cleanup_sources.py --data-dir data/raw
"""
import os
import shutil
import argparse
import yaml

CONFIG_PATH = "gdelt_config.yaml"

def load_trusted_sources(config_path=CONFIG_PATH):
    try:
        with open(config_path, 'r', encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh) or {}
        return set(cfg.get('trusted_sources', []))
    except Exception as e:
        print(f"[ERROR] Could not load config: {e}")
        return set()

def main(data_dir, dry_run):
    trusted = load_trusted_sources()
    if not trusted:
        print("No trusted sources found — aborting")
        return

    print(f"Trusted sources: {len(trusted)}")
    print(f"Scanning: {data_dir}\n")

    removed, kept = 0, 0

    for site_folder in os.listdir(data_dir):
        site_path = os.path.join(data_dir, site_folder)
        if not os.path.isdir(site_path):
            continue

        if site_folder in trusted:
            print(f"  [KEEP]   {site_folder}")
            kept += 1
        else:
            print(f"  [DELETE] {site_folder}")
            if not dry_run:
                shutil.rmtree(site_path)
            removed += 1

    print(f"\nDone: {kept} kept, {removed} {'would be ' if dry_run else ''}removed")
    if dry_run:
        print("Re-run without --dry-run to actually delete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--dry-run',  action='store_true',
                        help='Show what would be deleted without deleting')
    args = parser.parse_args()
    main(args.data_dir, args.dry_run)