import os
import numpy as np

# See if Cython is installed
try:
    from Cython.Build import cythonize
# Do nothing if Cython is not available
except ImportError:
    # Got to provide this function. Otherwise, poetry will fail
    def build(setup_kwargs):
        pass
# Cython is installed. Compile
else:
    from setuptools import Extension
    from setuptools.dist import Distribution
    from distutils.command.build_ext import build_ext

    # This function will be executed in setu`p.py:
    def build(setup_kwargs):
        # The file you want to compile
        extensions = [
            "nav_sim_modules/nav_components/mapping.pyx",
            "nav_sim_modules/nav_components/planning.pyx"
        ]

        # gcc arguments hack: enable optimizations
        os.environ['CFLAGS'] = '-O3'

        # Build
        setup_kwargs.update({
            'ext_modules': cythonize(
                extensions,
                language_level=3,
                language='c++',
                compiler_directives={'linetrace': True},
            ),
            # 'extra_compile_args': ['-fopenmp'],
            # 'extra_link_args': ['-fopenmp'],
            'cmdclass': {'build_ext': build_ext},
            'include_dirs': np.get_include()
        })