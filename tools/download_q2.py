import sys

sys.path.insert(0, "tools")

from remote_tool import download  # noqa: E402

if __name__ == "__main__":
    local = download("/home/gejw/mm/mm_v2/outputs/v2/q2", "outputs/v2")
    print("downloaded to", local)
