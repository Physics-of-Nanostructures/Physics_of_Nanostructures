import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='fnatools',
                 version='0.0.1',
                 description='General toolbox for FNA',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='https://github.com/Physics-of-Nanostructures/Physics_of_Nanostructures',
                 author='Physics-of-Nanostructures',
                 author_email='c.f.schippers@tue.nl',
                 packages=setuptools.find_packages(),
                 classifiers=(
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ),
                 )
