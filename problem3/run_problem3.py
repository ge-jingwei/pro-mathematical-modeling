"""Compatibility entry point for the requested unified command."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Ques3.ques3 import main

if __name__ == "__main__":
    main()
