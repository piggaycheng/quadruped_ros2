# How to install and use quadruped_ros2
## 安裝uv環境
```bash
# 安裝pipx
sudo apt update
sudo apt install pipx
pipx ensurepath
# 安裝uv
pipx install uv
```

## 安裝quadruped_ros2 python套件
```bash
uv pip install -r pyproject.toml --extra cpu
# 或者
uv pip install -r pyproject.toml --extra gpu
```
如果要安裝特定版本的torch，可以去修改`pyproject.toml`裡面的版本號


## Build quadruped_ros2
```bash
source .venv/bin/activate
python -m colcon build --packages-select quadruped_ros2 --symlink-install
```

## Run quadruped_ros2
```bash
source install/setup.bash
ros2 run quadruped_ros2 inference_node
```