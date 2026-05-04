"""Split bench2drive episodes into train/valid by Route ID (deterministic hash)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

EPISODE_RE = re.compile(
    r"^(?P<scenario>.+?)_Town(?P<town>\w+?)_Route(?P<route>\d+)_Weather(?P<weather>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="bench2drive root directory containing per-episode subdirs.",
    )
    parser.add_argument(
        "--valid_mod",
        type=int,
        default=20,
        help="1/valid_mod of unique Route IDs go to valid (default: 20 -> ~5%%).",
    )
    return parser.parse_args()


def parse_episode(name: str) -> dict | None:
    m = EPISODE_RE.match(name)
    return m.groupdict() if m else None


def is_valid_route(route_id: str, valid_mod: int) -> bool:
    """Deterministic by hashing route id; ~ 1/valid_mod fraction goes to valid."""
    h = hashlib.md5(route_id.encode()).hexdigest()
    return int(h, 16) % valid_mod == 0


def main() -> None:
    args = parse_args()

    out_path = args.src / "splits.json"

    episodes = sorted(p.name for p in args.src.iterdir() if p.is_dir() and EPISODE_RE.match(p.name))
    parsed: dict[str, dict] = {ep: parse_episode(ep) for ep in episodes}

    valid_routes = {
        ep for ep, info in parsed.items() if is_valid_route(info["route"], args.valid_mod)
    }
    train = sorted(ep for ep in parsed if ep not in valid_routes)
    valid = sorted(valid_routes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"train": train, "valid": valid}, f, indent=2)

    n_train_routes = len({parsed[ep]["route"] for ep in train})
    n_valid_routes = len({parsed[ep]["route"] for ep in valid})
    train_scenarios = Counter(parsed[ep]["scenario"] for ep in train)
    valid_scenarios = Counter(parsed[ep]["scenario"] for ep in valid)

    print(f"total episodes: {len(parsed)}")
    print(f"train: {len(train)} episodes / {n_train_routes} unique Route IDs")
    print(f"valid: {len(valid)} episodes / {n_valid_routes} unique Route IDs")
    print(
        f"valid scenario coverage: {len(valid_scenarios)} / {len(train_scenarios | valid_scenarios)} scenarios"
    )
    missing = sorted(set(train_scenarios) - set(valid_scenarios))
    if missing:
        print(f"  scenarios missing in valid: {missing}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
