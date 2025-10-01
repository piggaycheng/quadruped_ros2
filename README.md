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
python -m colcon build --packages-select quadruped --symlink-install
```

## Run quadruped_ros2
```bash
source install/setup.bash
ros2 run quadruped inference_node
```

## Launch joint state publisher
```bash
ros2 launch quadruped robot.launch.py
```

## Run ik_test_node
```bash
ros2 run quadruped ik_test_node
```

## Prepare URDF file
將urdf相關檔案都放在resource資料夾中，打包時會被包含進install/quadruped/share/quadruped/，結構如下
```quadruped_ros2
ros2_workspace
    └── src
        └── quadruped_ros2
            ├── resource
            │   ├── urdf
            │   │   └── go2.urdf
            │   └── meshes
            │       └── go2
            │           ├── body.stl
            │           ├── leg_link1.stl
            │           ├── leg_link2.stl
            │           └── leg_link3.stl
            └── ...
```
* urdf中的mesh路徑要改成`package://quadruped/meshes/go2/xxx.stl`
* 如果要啟動rviz2，要先source install/setup.bash，rviz2才能找到urdf檔案