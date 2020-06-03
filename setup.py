from distutils.core import setup

required_packages = ['numpy', 'scipy', 'h5py']  # , 'memory_profiler', 'pandas', 'vtk'
setup(
    name='MeshTransformation',
    version='0.9.0b5',
    description='''Farhad said :: blah blah ''',
    author='Farhad Daei',
    author_email='farhad.daei@helsinki.fi',
    license='MIT',
    packages=['DataSet'],
    install_requires=required_packages,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.6, <4',
    keywords='Mesh Transformation',
)
