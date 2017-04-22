#!/usr/bin/env python

from setuptools import setup

setup(
    name='smartmanuscript',
    version='0.2',
    description='Transcribe (digitize) handwritten manuscripts and digitizer pen input',
    author='Daniel Vorberg',
    author_email='dv@pks.mpg.de',
    url='https://github.com/antemons/smart-manuscript',
    packages=['smartmanuscript'],
    package_dir={'smartmanuscript': 'smartmanuscript'},
    scripts=["transcribe", "manuscript-writer"],
    include_package_data = True,
    long_description=""" """)
