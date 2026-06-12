# -*- coding: utf-8 -*-
"""
run_reconstruct_full.py
在 PyCharm 里直接运行：只需修改下方“配置区”的路径。
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ========================== 配置区（按需修改） ==========================
# MipMapEngine 的 ReconstructFull 可执行文件
ENGINE_EXE = r"D:\zyh\Win1\WinfromMyMap\WinfromMyMap\bin\Debug\net8.0-windows\3d_sdk_2\reconstruct_full_engine.exe"

# 你的任务 JSON（用我之前给你的脚本生成的即可）
TASK_JSON = r"D:\testhcl\reconstruct_full_config.json"

# 是否把 image_meta_data[].path 统一写成绝对路径（强烈建议 True）
FORCE_ABS_PATHS = True

# 是否把引擎的工作目录（进程 cwd）设为 exe 所在目录（通常更稳妥）
SET_CWD_TO_ENGINE_DIR = True

# 是否把控制台输出另存为日志文件（保存在 JSON 同级目录）
SAVE_LOG = True
# =====================================================================


def _ensure_abs_paths(task_json_path: Path, force_abs: bool) -> bool:
    """
    确保 image_meta_data[].path 为绝对路径。
    返回：是否有修改
    """
    if not force_abs:
        return False

    with task_json_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    images = cfg.get("image_meta_data", [])
    if not isinstance(images, list):
        raise ValueError("task_json 中缺少 image_meta_data 数组。")

    base = task_json_path.parent.resolve()
    changed = False

    for im in images:
        p = im.get("path")
        if not isinstance(p, str) or not p:
            continue
        if os.path.isabs(p):
            # 规范化分隔符
            newp = str(Path(p).resolve())
        else:
            # 相对 JSON 的目录转为绝对路径
            newp = str((base / p).resolve())
        newp_norm = newp.replace("\\", "/")
        if p != newp_norm:
            im["path"] = newp_norm
            changed = True

    if changed:
        with task_json_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    return changed


def run_reconstruct_full(engine_exe: Path, task_json: Path,
                         set_cwd_to_engine_dir: bool = True,
                         save_log: bool = True) -> int:
    if not engine_exe.exists():
        raise FileNotFoundError(f"找不到引擎可执行文件：{engine_exe}")
    if not task_json.exists():
        raise FileNotFoundError(f"找不到任务 JSON：{task_json}")

    cmd = [
        str(engine_exe),
        "-reconstruct_type", "0",     # ReconstructFull 固定为 0
        "-task_json", str(task_json)
    ]

    cwd = engine_exe.parent if set_cwd_to_engine_dir else None

    print("启动命令：", " ".join(cmd))
    print("工作目录：", cwd if cwd else Path().resolve())

    log_fp = None
    if save_log:
        log_name = f"reconstruct_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_fp = (task_json.parent / log_name).open("w", encoding="utf-8")
        print(f"日志输出：{task_json.parent / log_name}")

    # 流式读取输出，实时打印
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd) if cwd else None,
        shell=False,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        if log_fp:
            log_fp.write(line)

    proc.wait()
    if log_fp:
        log_fp.write(f"\n=== Process exited with code {proc.returncode} ===\n")
        log_fp.close()

    print(f"\n进程退出码：{proc.returncode}")
    return proc.returncode


def main():
    engine = Path(ENGINE_EXE)
    task = Path(TASK_JSON)

    if FORCE_ABS_PATHS:
        changed = _ensure_abs_paths(task, True)
        if changed:
            print("已将 image_meta_data[].path 统一转换为绝对路径。")

    code = run_reconstruct_full(
        engine_exe=engine,
        task_json=task,
        set_cwd_to_engine_dir=SET_CWD_TO_ENGINE_DIR,
        save_log=SAVE_LOG
    )

    if code != 0:
        print("重建失败（非零退出码）。请检查日志与 JSON 配置。")
        sys.exit(code)
    else:
        print("重建完成。")


if __name__ == "__main__":
    main()
