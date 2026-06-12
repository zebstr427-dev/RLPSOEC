# -*- coding: utf-8 -*-
"""
mipmap_config_builder.py
在 PyCharm 里直接运行：修改下方“配置区”变量，然后 Run。
生成的 JSON 结构用于 MipMapEngine reconstruct_full。
"""

import json
import os
from pathlib import Path

# ========================== 配置区（按需修改） ==========================
IMAGES_FOLDER = r"D:\testhcl\Image"   # 图片所在文件夹
OUTPUT_JSON   = r"D:\testhcl\reconstruct_full_config.json"  # 输出 JSON 完整路径

GROUP_NAME    = "camera_1"        # 目前全部在同一组
WORKING_DIR   = r"D:\testhcl\wkd" # 保持你原来的 working_dir
GDAL_FOLDER   = r"D:\zyh\Win1\WinfromMyMap\WinfromMyMap\bin\Debug\net8.0-windows\data2"  # 保持原来的 gdal_folder

USE_ABSOLUTE_PATHS = False        # 写绝对路径；若为 False 写相对 OUTPUT_JSON 的相对路径
# =====================================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _list_images(folder: Path):
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"找不到图片文件夹：{folder}")
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: p.name.lower())
    if not files:
        raise RuntimeError("该文件夹未发现支持的图片：jpg/jpeg/png/tif/tiff/bmp")
    return files


def _to_rel_or_abs(path: Path, base: Path, use_abs: bool) -> str:
    if use_abs:
        return str(path.resolve()).replace("\\", "/")
    try:
        rel = os.path.relpath(path.resolve(), start=base.resolve())
        return rel.replace("\\", "/")
    except Exception:
        # 兜底：实在算不了相对路径就写绝对路径
        return str(path.resolve()).replace("\\", "/")


def build_config(images_folder: Path,
                 output_json_path: Path,
                 group_name: str,
                 working_dir: str,
                 gdal_folder: str,
                 use_absolute_paths: bool = False) -> dict:
    images = _list_images(images_folder)

    base_dir = output_json_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    image_meta = []
    for i, img in enumerate(images, start=1):
        image_meta.append({
            "id": i,
            "path": _to_rel_or_abs(img, base_dir, use_absolute_paths),
            "group": group_name
        })

    cfg = {
        "license_id": 9200,
        "working_dir": working_dir,
        "gdal_folder": gdal_folder,
        "input_image_type": 1,
        "resolution_level": 3,

        # 输出项（保持与你之前示例一致；需要可自行改）
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

        # 坐标系：DJI P1 默认经纬度坐标为 WGS84
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


def main():
    images_folder = Path(IMAGES_FOLDER)
    output_json_path = Path(OUTPUT_JSON)

    cfg = build_config(
        images_folder=images_folder,
        output_json_path=output_json_path,
        group_name=GROUP_NAME,
        working_dir=WORKING_DIR,
        gdal_folder=GDAL_FOLDER,
        use_absolute_paths=USE_ABSOLUTE_PATHS
    )

    save_config(cfg, output_json_path)
    print(f"已生成：{output_json_path}")
    print(f"图片数量：{len(cfg['image_meta_data'])}")
    if not USE_ABSOLUTE_PATHS:
        print("提示：采用相对路径（相对输出 JSON 所在目录）。如需绝对路径，将 USE_ABSOLUTE_PATHS 设为 True。")


if __name__ == "__main__":
    main()
