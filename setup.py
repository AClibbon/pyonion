from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(os.path.join(here, 'requirements-dev.txt')) as f:
    dev_reqs = f.read().splitlines()
with open(os.path.join(here, 'requirements-spark.txt')) as f:
    spark_reqs = f.read().splitlines()



setup(
    name="pyonion",
    use_scm_version = {
        "root": "..",
        "relative_to": __file__,
        "local_scheme": "node-and-timestamp"
    },
    setup_requires=['setuptools_scm'],
    author="KAPUK",
    author_email="alex.clibbon@kantar.com",
    description="A minimal implementation of the ONe Instance ONly algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],
    include_package_data=False,
    url="https://github.com/AClibbon/pyonion",
    classifiers=[
        "Programming Language :: Python :: 3",
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Topic :: Text Processing :: Linguistic'
    ],
    python_requires=">=3.7",

)
