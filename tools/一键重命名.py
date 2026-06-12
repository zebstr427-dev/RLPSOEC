import os
import subprocess

# ① 这里填你的 MTS 文件所在文件夹
input_folder = r"E:\PRIVATE\AVCHD\BDMV\STREAM"
output_folder = os.path.join(r"D:\zyh\skate\摄影素材\新建文件夹")

# ② 这里填 ffmpeg.exe 的完整路径（重点！！！）
ffmpeg_path = r"D:\app\ffmpeg\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".mts"):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".mp4"
        output_path = os.path.join(output_folder, output_filename)

        command = [
            ffmpeg_path,   # ← 不再是 "ffmpeg"，而是绝对路径
            "-i", input_path,
            "-c:v", "copy",
            "-c:a", "aac",
            output_path
        ]

        print(f"正在转换: {filename} → {output_filename}")

        # 加上 check=True，方便报错时看到原因
        subprocess.run(command, check=True)

print("\n🎉 全部转换完成！视频已保存到：", output_folder)
