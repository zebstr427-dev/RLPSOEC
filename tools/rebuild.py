import json
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List

def _run_exiftool(img: Path) -> dict:
    """
    调用 exiftool 读取所有元数据并以 JSON 返回。
    -n: 使用数值原始值
    -j: JSON 输出
    -G1: 带群组（如 MakerNotes）
    -s: 使用简短标签名
    """
    cmd = ["exiftool", "-n", "-j", "-G1", "-s", str(img)]
    out = subprocess.check_output(cmd)
    data = json.loads(out)[0]
    return data

def _parse_gimbal_degree(val) -> Optional[Tuple[int, int, int]]:
    """
    把 exiftool 返回的 GimbalDegree 字段解析为 (a, b, c) 三元组。
    可能的情况：
      - 字符串: '283,-700,-900'
      - 列表: [283, -700, -900]
    """
    if val is None:
        return None
    if isinstance(val, str):
        parts = [p.strip() for p in val.replace(";", ",").split(",") if p.strip()]
        if len(parts) != 3:
            return None
        return tuple(int(float(p)) for p in parts)  # 兼容 '283.0' 之类
    if isinstance(val, list):
        if len(val) != 3:
            return None
        return tuple(int(float(x)) for x in val)
    return None

def read_dji_gimbal_degree(image_path: str) -> Optional[Tuple[int, int, int]]:
    """
    读取大疆 JPG 的 Gimbal Degree 值。
    返回 (v1, v2, v3)，例如 (283, -700, -900)。
    """
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(image_path)
    meta = _run_exiftool(img)

    # 兼容不同键名（有的版本是 'GimbalDegree'，有的是 'Gimbal Degree'，或带分组名）
    possible_keys: List[str] = []
    for k in meta.keys():
        lk = k.lower()
        if "gimbaldegree" in lk.replace(" ", ""):
            possible_keys.append(k)

    val = None
    for k in possible_keys:
        val = meta.get(k)
        if val is not None:
            break

    return _parse_gimbal_degree(val)

if __name__ == "__main__":
    # 示例：替换为你的 P1 照片路径
    path = r"/path/to/your/DJI_P1_photo.jpg"
    deg = read_dji_gimbal_degree(path)
    if deg is None:
        print("未找到 Gimbal Degree 字段。")
    else:
        a, b, c = deg
        print(f"Gimbal Degree: {deg}")
        # 如果你只关心值为 -700 的那个分量，可这么判断：
        which = ("#1", "#2", "#3")[deg.index(-700)] if -700 in deg else "（无 -700）"
        print(f"含 -700 的分量：{which}")
