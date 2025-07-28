import subprocess
import sys

def install_ffmpeg():
  print("Starting FFmpeg installation...")
  
  subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
  subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])
  
  try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
    print("FFmpeg installed successfully.")
    
  except subprocess.CalledProcessError as e:
    print(f"Error installing FFmpeg via pip: {e}")

  try:
    subprocess.check_call([
      "wget", "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
      "-O", "/tmp/ffmpeg.tar.xz"
    ])
    
    subprocess.check_call(["tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp"])
    
    result = subprocess.run(
      ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
      text=True,
    )
    
    ffmeg_path = result.stdout.strip()
    
    subprocess.check_call(["cp", ffmeg_path, "/usr/local/bin/ffmpeg"])
    subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])
    
    print("Installed FFmpeg from binary successfully.")
    
  except subprocess.CalledProcessError as e:
    print(f"Error installing static FFmpeg from binary: {e}")
  
  try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    print("FFmpeg version:")
    print(result.stdout)
    return True
  
  except (subprocess.CalledProcessError, FileNotFoundError) as e:
    print(f"FFmpeg installation verification failed: {e}")
    return False