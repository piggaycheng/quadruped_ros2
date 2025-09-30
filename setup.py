import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'quadruped'


def get_data_files():
    """Get all data files for the package."""
    data_files = []

    # Add basic package files
    data_files.append(('share/ament_index/resource_index/packages',
                      ['resource/' + package_name]))
    data_files.append(('share/' + package_name, ['package.xml']))

    # Add launch files
    launch_files = glob('launch/*.launch.py')
    if launch_files:
        data_files.append(('share/' + package_name + '/launch', launch_files))

    # Add all resource files recursively (excluding package index file)
    resource_dirs = set()

    for root, dirs, files in os.walk('resource'):
        for file in files:
            # Skip the package index file as it's handled separately
            if root == 'resource' and file == package_name:
                continue
                
            file_path = os.path.join(root, file)
            # Calculate the relative directory path for installation
            rel_dir = os.path.relpath(root, 'resource')
            if rel_dir == '.':
                install_dir = 'share/' + package_name
            else:
                install_dir = 'share/' + package_name + '/' + rel_dir
            resource_dirs.add((install_dir, file_path))

    # Group files by installation directory
    dir_files = {}
    
    # Process all resource files
    for install_dir, file_path in resource_dirs:
        if install_dir not in dir_files:
            dir_files[install_dir] = []
        dir_files[install_dir].append(file_path)

    # Add grouped files to data_files
    for install_dir, files in dir_files.items():
        data_files.append((install_dir, files))

    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=get_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yu',
    maintainer_email='piggaycheng123@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_node = quadruped.inference_node:main',
            'ik_test_node = quadruped.ik_test_node:main'
        ],
    },
)
