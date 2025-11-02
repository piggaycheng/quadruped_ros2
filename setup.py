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

    # Add all resource files recursively
    if os.path.exists('resource'):
        for root, dirs, files in os.walk('resource'):
            # Don't install the package index file
            if root == 'resource' and package_name in files:
                files.remove(package_name)
            
            install_dir = os.path.join('share', package_name, os.path.relpath(root, '.'))
            if files:
                data_files.append((install_dir, [os.path.join(root, f) for f in files]))

    # Add config files recursively
    if os.path.exists('config'):
        for root, dirs, files in os.walk('config'):
            install_dir = os.path.join('share', package_name, os.path.relpath(root, '.'))
            if files:
                data_files.append((install_dir, [os.path.join(root, f) for f in files]))

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
