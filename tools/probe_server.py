import sys

sys.path.insert(0, "tools")

from remote_tool import execute  # noqa: E402

COMMAND = (
    "wc -l /home/gejw/mm/logs/q2_v2.log; echo '--- tail ---'; tail -n 30 /home/gejw/mm/logs/q2_v2.log; "
    "echo '--- gpu ---'; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; "
    "echo '--- proc ---'; ps aux | grep -E 'run_q2_v2|python -m' | grep -v grep | head -5"
)

if __name__ == "__main__":
    sys.exit(execute(COMMAND, workdir=None, conda_env=None))
