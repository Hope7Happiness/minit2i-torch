#!/usr/bin/env python
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("usage: main.py {train,eval} [args...]")
        print()
        print("commands:")
        print("  train   run training; forwards args to mini_t2i.train")
        print("  eval    run evaluation; forwards args to mini_t2i.eval_pipeline")
        return

    command = sys.argv[1]
    rest = sys.argv[2:]

    if command == "train":
        from mini_t2i.train import main as train_main

        sys.argv = [sys.argv[0], *rest]
        train_main()
        return

    if command == "eval":
        from mini_t2i.eval_pipeline import main as eval_main

        sys.argv = [sys.argv[0], *rest]
        eval_main()
        return

    raise ValueError(f"unknown command: {command}")


if __name__ == "__main__":
    main()
