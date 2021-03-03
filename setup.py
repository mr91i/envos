from setuptools import setup, find_packages

setup(
    name='envos',
    version='0.1.0',
    license='GNU General Public License v3.0',
    description='Envelope observation simulator',
    install_requires=["numpy", "matplotlib", "pandas", "astropy", "scipy", "dataclasses"],
    author='Shoji Mori',
    author_email='shoji9m@gmail.com',
    url='https://github.com/mr91i/envos',
    packages=find_packages(),
#    package_dir={'': 'envos'},
    python_requires='>=3.6, <4',

)
