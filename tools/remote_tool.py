import argparse
import os
import shlex
import stat
import sys
import time

import paramiko

HOST = os.environ.get("REMOTE_HOST", "172.22.226.86")
PORT = int(os.environ.get("REMOTE_PORT", "22"))
USERNAME = os.environ.get("REMOTE_USER", "gejw")
PASSWORD = os.environ.get("REMOTE_PASSWORD", "gejingwei")
KEY_PATH = os.environ.get("REMOTE_KEY", "")
ROOT_PATH = os.environ.get("REMOTE_ROOT", "/home/gejw/mm")
CONDA_NAME = os.environ.get("REMOTE_CONDA_ENV", "gejw")


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs = dict(hostname=HOST, port=PORT, username=USERNAME, timeout=20)
    if KEY_PATH:
        kwargs["key_filename"] = KEY_PATH
    else:
        kwargs["password"] = PASSWORD
    client.connect(**kwargs)
    return client


def _sftp_mkdirs(sftp, remote_dir):
    parts = remote_dir.split("/")
    path = "/" if remote_dir.startswith("/") else ""
    for part in parts:
        if not part:
            continue
        path = path.rstrip("/") + "/" + part
        try:
            sftp.stat(path)
        except IOError:
            sftp.mkdir(path)


def file_upload(path, remote_dir=ROOT_PATH, remote_name=None):
    """Upload a local file or directory tree into remote_dir."""
    client = connect()
    sftp = client.open_sftp()
    try:
        name = remote_name or os.path.basename(os.path.normpath(path))
        remote_path = remote_dir.rstrip("/") + "/" + name
        if os.path.isdir(path):
            _sftp_mkdirs(sftp, remote_path)
            for root, _dirs, files in os.walk(path):
                rel = os.path.relpath(root, path)
                cur = remote_path if rel == "." else remote_path + "/" + rel.replace("\\", "/")
                _sftp_mkdirs(sftp, cur)
                for f in files:
                    sftp.put(os.path.join(root, f), cur + "/" + f)
        else:
            _sftp_mkdirs(sftp, remote_dir)
            sftp.put(path, remote_path)
        return remote_path
    finally:
        sftp.close()
        client.close()


def _shell(client, command, pty=False, log_file=None):
    chan = client.get_transport().open_session()
    if pty:
        chan.get_pty()
    chan.exec_command("bash -lc %s" % shlex.quote(command))
    chunks = []
    while True:
        while chan.recv_ready():
            data = chan.recv(65536)
            if not data:
                break
            chunks.append(data)
            text = data.decode("utf-8", "replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            if log_file:
                log_file.write(text)
        if chan.exit_status_ready() and not chan.recv_ready():
            break
        time.sleep(0.05)
    code = chan.recv_exit_status()
    text = b"".join(chunks).decode("utf-8", "replace")
    return code, text


def execute(command, workdir=None, conda_env=CONDA_NAME, log=None):
    """Run a command on the server, stream output, return the exit code.

    The command runs with bash login shell in workdir and inside conda_env.
    Remote stdout is printed in real time and copied into log when given.
    """
    if conda_env:
        command = "conda run --no-capture-output -n %s -- %s" % (
            shlex.quote(conda_env), command)
    if workdir:
        command = "cd %s && %s" % (shlex.quote(workdir), command)
    client = connect()
    try:
        log_file = open(log, "w", encoding="utf-8", errors="replace", newline="") if log else None
        try:
            code, _ = _shell(client, command, pty=True, log_file=log_file)
        finally:
            if log_file:
                log_file.close()
        return code
    finally:
        client.close()


def run_background(command, workdir=None, conda_env=CONDA_NAME, log=None, root=ROOT_PATH):
    """Start a command on the server in the background with nohup.

    Output is written to the remote log file; the pid is stored in
    <log>.pid so job_status can query it later. Returns (pid, remote_log).
    """
    log = log or (root.rstrip("/") + "/logs/run_" + time.strftime("%Y%m%d_%H%M%S") + ".log")
    if not log.startswith("/"):
        log = root.rstrip("/") + "/" + log
    if conda_env:
        command = "conda run --no-capture-output -n %s -- %s" % (
            shlex.quote(conda_env), command)
    if workdir:
        command = "cd %s && %s" % (shlex.quote(workdir), command)
    logdir = "/".join(log.split("/")[:-1]) or "/"
    wrap = "mkdir -p %s; nohup bash -lc %s > %s 2>&1 </dev/null & echo $!" % (
        shlex.quote(logdir), shlex.quote(command), shlex.quote(log))
    client = connect()
    try:
        code, text = _shell(client, wrap, pty=False)
        pid = None
        for line in reversed(text.strip().splitlines()):
            if line.strip().isdigit():
                pid = int(line.strip())
                break
        if code != 0 or pid is None:
            raise RuntimeError("failed to start background job (code=%s): %s" % (code, text.strip()))
        sftp = client.open_sftp()
        try:
            with sftp.open(log + ".pid", "w") as f:
                f.write(str(pid) + "\n")
        finally:
            sftp.close()
        return pid, log
    finally:
        client.close()


def job_status(log, tail=40, root=ROOT_PATH):
    """Print whether the background job is running and tail its log.

    Returns True when still running, False when finished or unknown.
    """
    if not log.startswith("/"):
        log = root.rstrip("/") + "/" + log
    client = connect()
    try:
        pid = None
        sftp = client.open_sftp()
        try:
            with sftp.open(log + ".pid", "r") as f:
                data = f.read()
            pid = (data.decode("utf-8", "replace") if isinstance(data, bytes) else data).strip()
        except IOError:
            pid = None
        finally:
            sftp.close()
        if pid and pid.isdigit():
            snippet = ("if kill -0 %s 2>/dev/null; then echo '[STATUS] RUNNING (pid %s)'; "
                       "else echo '[STATUS] FINISHED (pid %s)'; fi; "
                       "echo '--- tail %s ---'; tail -n %d %s 2>&1" %
                       (pid, pid, pid, log, tail, shlex.quote(log)))
        else:
            snippet = ("echo '[STATUS] no pid file (%s.pid)'; "
                       "echo '--- tail %s ---'; tail -n %d %s 2>&1" %
                       (shlex.quote(log), log, tail, shlex.quote(log)))
        code, text = _shell(client, snippet, pty=False)
        running = "RUNNING" in text and code == 0
        return running
    finally:
        client.close()


def _download_rec(sftp, remote_path, local_path):
    attr = sftp.stat(remote_path)
    if stat.S_ISDIR(attr.st_mode):
        os.makedirs(local_path, exist_ok=True)
        for name in sftp.listdir(remote_path):
            _download_rec(sftp, remote_path.rstrip("/") + "/" + name,
                          os.path.join(local_path, name))
    else:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        sftp.get(remote_path, local_path)


def download(remote, local):
    """Download a remote file or directory tree into the local folder."""
    client = connect()
    sftp = client.open_sftp()
    try:
        name = os.path.basename(remote.rstrip("/"))
        _download_rec(sftp, remote, os.path.join(local, name))
        return os.path.join(local, name)
    finally:
        sftp.close()
        client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="remote_tool.py",
        description="Upload files, run training on the lab server via SSH, "
                    "stream logs, and download results.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_upload = sub.add_parser("upload", help="Upload a local file or folder to the server")
    p_upload.add_argument("path", help="Local file or folder")
    p_upload.add_argument("--remote-dir", default=ROOT_PATH,
                          help="Remote destination directory (default: %(default)s)")
    p_upload.add_argument("--remote-name", default=None,
                          help="Remote file/folder name (default: local basename)")
    p_upload.set_defaults(run=lambda a: print("uploaded:", file_upload(
        a.path, a.remote_dir, a.remote_name)))

    p_run = sub.add_parser("run", help="Run a command on the server and stream its output")
    p_run.add_argument("--cmd", required=True, help="Command to run, e.g. python train.py")
    p_run.add_argument("--workdir", default=None,
                       help="Remote working directory (default: %(default)s)")
    p_run.add_argument("--conda", default=CONDA_NAME,
                       help="Remote conda env name; pass empty string to disable")
    p_run.add_argument("--log", default=None,
                       help="Local file to save the command log")
    p_run.set_defaults(run=lambda a: sys.exit(execute(
        a.cmd, a.workdir, a.conda if a.conda else None, a.log)))

    p_bg = sub.add_parser("run-bg", help="Start a command on the server in the background")
    p_bg.add_argument("--cmd", required=True, help="Command to run, e.g. python train.py")
    p_bg.add_argument("--workdir", default=ROOT_PATH,
                      help="Remote working directory (default: %(default)s)")
    p_bg.add_argument("--conda", default=CONDA_NAME,
                      help="Remote conda env name; pass empty string to disable")
    p_bg.add_argument("--log", default=None,
                      help="Remote log path; pid is saved to <log>.pid "
                           "(default: logs/run_<timestamp>.log under remote root)")
    p_bg.set_defaults(run=lambda a: print("started pid=%s log=%s" % run_background(
        a.cmd, a.workdir, a.conda if a.conda else None, a.log)))

    p_st = sub.add_parser("status", help="Check a background job started by run-bg")
    p_st.add_argument("--log", required=True,
                      help="Remote log path (same value passed to run-bg --log)")
    p_st.add_argument("--tail", type=int, default=40, help="Lines of log to show")
    p_st.add_argument("--workdir", default=ROOT_PATH,
                      help="Remote root used to resolve a relative --log")
    p_st.set_defaults(run=lambda a: sys.exit(0 if job_status(a.log, a.tail, a.workdir) else 1))

    p_download = sub.add_parser("download", help="Download a remote file or folder")
    p_download.add_argument("remote", help="Remote path (file or directory)")
    p_download.add_argument("local", help="Local destination folder")
    p_download.set_defaults(run=lambda a: print("downloaded:", download(a.remote, a.local)))

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
