# -*- coding: utf-8 -*-
"""
one_click_reconstruct_full.py
在 PyCharm 中直接运行：
1) 修改下方“配置区”的路径与参数
2) 直接 Run
功能：遍历图片 -> 生成 reconstruct_full 所需 JSON -> 调用引擎执行
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ========================== 配置区（按需修改） ==========================
# 引擎可执行文件路径
ENGINE_EXE = r"D:\zyh\Win1\WinfromMyMap\WinfromMyMap\bin\Debug\net8.0-windows\3d_sdk_2\reconstruct_full_engine.exe"

# 图片文件夹（当前全部放同一组）
IMAGES_FOLDER = r"D:\testhcl\Image"

# 输出 JSON 完整路径
OUTPUT_JSON   = r"D:\testhcl\reconstruct_full_config.json"

# 分组与基础参数
GROUP_NAME    = "camera_1"
WORKING_DIR   = r"D:\testhcl\wkd"
GDAL_FOLDER   = r"D:\zyh\Win1\WinfromMyMap\WinfromMyMap\bin\Debug\net8.0-windows\data2"

# 新增字段：2D切片保存类型（按引擎定义设置正确值；此处默认 0）
TILE_2D_SAVE_TYPE = 1

# 路径与扫描选项
USE_ABSOLUTE_IMAGE_PATHS = True      # 建议 True：写入图片绝对路径
RECURSIVE_SCAN = False               # 是否递归扫描子文件夹
SET_CWD_TO_ENGINE_DIR = True         # 以引擎目录为工作目录启动
SAVE_LOG = True                      # 保存引擎输出到日志文件
# =====================================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path, recursive: bool):
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"找不到图片文件夹：{folder}")
    if recursive:
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: p.name.lower())
    if not files:
        raise RuntimeError("该文件夹未发现支持的图片：jpg/jpeg/png/tif/tiff/bmp")
    return files


def to_rel_or_abs(path: Path, base: Path, use_abs: bool) -> str:
    if use_abs:
        return str(path.resolve()).replace("\\", "/")
    try:
        rel = os.path.relpath(path.resolve(), start=base.resolve())
        return rel.replace("\\", "/")
    except Exception:
        # 兜底：无法计算相对路径时使用绝对路径
        return str(path.resolve()).replace("\\", "/")


def build_config(images_folder: Path,
                 output_json_path: Path,
                 group_name: str,
                 working_dir: str,
                 gdal_folder: str,
                 tile_2d_save_type,
                 use_absolute_paths: bool,
                 recursive_scan: bool) -> dict:
    images = list_images(images_folder, recursive_scan)
    base_dir = output_json_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    image_meta = []
    for i, img in enumerate(images, start=1):
        image_meta.append({
            "id": i,
            "path": to_rel_or_abs(img, base_dir, use_absolute_paths),
            "group": group_name
        })

    cfg = {
        "license_id": 9200,
        "working_dir": working_dir,
        "gdal_folder": gdal_folder,
        "input_image_type": 1,
        "resolution_level": 3,

        # 输出控制（可按需修改）
        "output_block_change_xml": True,
        "generate_osgb": True,
        "generate_3d_tiles": True,
        "generate_las": True,
        "generate_pc_osgb": False,
        "generate_pc_pnts": False,
        "generate_pc_ply": False,
        "generate_obj": True,
        "generate_ply": False,
        "generate_geotiff": True,
        "generate_tile_2D": True,
        "generate_2D_from_3D_model": True,

        # 新增字段：2D切片保存类型（请按引擎定义设置值）
        "tile_2d_save_type": tile_2d_save_type,

        # 坐标系：DJI P1 默认经纬度为 WGS84
        "coordinate_system": {
            "type": 2,
            "type_name": "Geographic",
            "label": "WGS 84",
            "epsg_code": 4326
        },

        "image_meta_data": image_meta
    }

    return cfg


def save_config(cfg: dict, output_json_path: Path):
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def run_reconstruct_full(engine_exe: Path, task_json: Path,
                         set_cwd_to_engine_dir: bool = True,
                         save_log: bool = True) -> int:
    if not engine_exe.exists():
        raise FileNotFoundError(f"找不到引擎可执行文件：{engine_exe}")
    if not task_json.exists():
        raise FileNotFoundError(f"找不到任务 JSON：{task_json}")

    cmd = [
        str(engine_exe),
        "-reconstruct_type", "0",   # ReconstructFull
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
    engine_exe = Path(ENGINE_EXE)
    images_folder = Path(IMAGES_FOLDER)
    output_json_path = Path(OUTPUT_JSON)

    # 1) 生成 JSON
    cfg = build_config(
        images_folder=images_folder,
        output_json_path=output_json_path,
        group_name=GROUP_NAME,
        working_dir=WORKING_DIR,
        gdal_folder=GDAL_FOLDER,
        tile_2d_save_type=TILE_2D_SAVE_TYPE,
        use_absolute_paths=USE_ABSOLUTE_IMAGE_PATHS,
        recursive_scan=RECURSIVE_SCAN
    )
    save_config(cfg, output_json_path)
    print(f"JSON 已生成：{output_json_path}")
    print(f"图片数量：{len(cfg['image_meta_data'])}")
    if USE_ABSOLUTE_IMAGE_PATHS:
        print("已写入图片绝对路径。")

    # 2) 调用引擎执行
    code = run_reconstruct_full(
        engine_exe=engine_exe,
        task_json=output_json_path,
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
