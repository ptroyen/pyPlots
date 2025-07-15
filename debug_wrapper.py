#!/usr/bin/env python3
import sys
import traceback

if __name__ == "__main__":
    try:
        from pyplots.cli import main
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)