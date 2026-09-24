"""Run Question 1 from any working directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Ques1.src.run_analysis import main

if __name__ == "__main__":
    main()
