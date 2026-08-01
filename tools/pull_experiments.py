#!/usr/bin/env python3
"""
在 B 主机上运行：通过 SSH/rsync 从 A 主机拉取 experiments 目录内容。

默认：
  A: root@10.99.1.112:59889
  A目录: /root/autodl-tmp/pro/UIE3_workspace/UIE3_VN/experiments
  B目录: /public/home/hnust15874739861/pro/UIE3_VN/experiments

默认不删除 B 中额外文件；使用 --delete 才执行镜像删除。
SSH 密码不会保存在脚本中，rsync/ssh 会在终端中交互询问。
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_HOST = "10.99.1.112"
DEFAULT_PORT = 59889
DEFAULT_USER = "root"
DEFAULT_SOURCE = "/root/autodl-tmp/pro/UIE3_workspace/UIE3_VN/experiments"
DEFAULT_DESTINATION = "/public/home/hnust15874739861/pro/UIE3_VN/experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 B 主机上通过 rsync 从 A 主机拉取 experiments 目录。"
    )
    parser.add_argument("-H", "--host", default=DEFAULT_HOST, help="A 主机地址")
    parser.add_argument("-P", "--port", type=int, default=DEFAULT_PORT, help="A 主机 SSH 端口")
    parser.add_argument("-u", "--user", default=DEFAULT_USER, help="A 主机 SSH 用户")
    parser.add_argument("-s", "--source", default=DEFAULT_SOURCE, help="A 主机源目录")
    parser.add_argument("-d", "--destination", default=DEFAULT_DESTINATION, help="B 主机目标目录")
    parser.add_argument("-i", "--identity", help="SSH 私钥文件")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="删除 B 中在 A 不存在的文件，使目标目录与源目录镜像一致",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="正式传输前，将 B 的现有目标目录重命名备份",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="仅预演，不实际修改文件",
    )
    return parser.parse_args()


def fail(message: str, exit_code: int = 2) -> None:
    print(f"错误：{message}", file=sys.stderr)
    raise SystemExit(exit_code)


def main() -> int:
    args = parse_args()

    if not 1 <= args.port <= 65535:
        fail("SSH 端口必须在 1 到 65535 之间")
    if shutil.which("rsync") is None:
        fail("未安装 rsync")
    if shutil.which("ssh") is None:
        fail("未安装 ssh")

    source = args.source.rstrip("/")
    destination = Path(args.destination.rstrip("/")).expanduser()

    if not source:
        fail("源目录不能为空")
    if str(destination) in {"", "."}:
        fail("目标目录无效")

    identity: Path | None = None
    if args.identity:
        identity = Path(args.identity).expanduser()
        if not identity.is_file():
            fail(f"SSH 私钥不存在：{identity}")

    if not args.dry_run:
        if args.backup and destination.exists():
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = destination.with_name(f"{destination.name}_backup_{timestamp}")
            print(f"备份 B 端目录：{destination} -> {backup}")
            destination.rename(backup)
        destination.mkdir(parents=True, exist_ok=True)

    ssh_parts = [
        "ssh",
        "-p",
        str(args.port),
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=6",
    ]
    if identity is not None:
        ssh_parts.extend(["-i", str(identity)])

    rsync_args = [
        "rsync",
        "-aHh",
        "--partial",
        "--append-verify",
        "--no-owner",
        "--no-group",
        "--protect-args",
        "--info=progress2",
    ]
    if args.delete:
        rsync_args.append("--delete-delay")
    if args.dry_run:
        rsync_args.extend(["--dry-run", "--itemize-changes"])

    remote_source = f"{args.user}@{args.host}:{source}/"
    local_destination = f"{destination}/"
    rsync_args.extend(["-e", shlex.join(ssh_parts), remote_source, local_destination])

    mode = "预演" if args.dry_run else "正式传输"
    deletion = "镜像删除已启用" if args.delete else "保留 B 端额外文件"
    print(f"源：{remote_source}")
    print(f"目标：{local_destination}")
    print(f"模式：{mode}；{deletion}\n")
    print("执行：", shlex.join(rsync_args), "\n")

    try:
        completed = subprocess.run(rsync_args, check=False)
    except KeyboardInterrupt:
        print("\n用户中断。已传输的临时数据可由下次 rsync 继续利用。", file=sys.stderr)
        return 130
    except OSError as exc:
        print(f"无法启动 rsync：{exc}", file=sys.stderr)
        return 1

    if completed.returncode != 0:
        print(f"\n同步失败，rsync 返回码：{completed.returncode}", file=sys.stderr)
        return completed.returncode

    if args.dry_run:
        print("\n预演完成，未修改任何文件。")
    else:
        print("\n同步完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
