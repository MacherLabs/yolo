from setuptools import setup

setup(
    name='yolo',

    version='1.1.0',
    description='Yolo general object detection',
    url='http://demo.vedalabs.in/',

    # Author details
    author='Atinderpal Singh',
    author_email='atinderpalap@gmail.com',

    license='Commercial',

    packages=['yolo'],
    package_data={
        'yolo': ['weights/*', 'cfg/*'],
    },

    install_requires=['cython'],
    zip_safe=False
    )
