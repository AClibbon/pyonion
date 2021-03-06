from setuptools import setup
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
    use_scm_version={
        "local_scheme": "no-local-version",
        "version_scheme": "guess-next-dev"
    },
    setup_requires=['setuptools_scm'],
    author="KAPUK",
    author_email="alex.clibbon@kantar.com",
    description="A minimal implementation of the ONe Instance ONly algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['pyonion'],
    install_requires=[],
    extras_require={
        "dev": dev_reqs,
        "spark": spark_reqs
    },
    package_data={
        'pyonion': ['data/*'],
    },
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
