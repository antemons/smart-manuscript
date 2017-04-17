#!/usr/bin/env python

from setuptools import setup

setup(
    name='smart_manuscript',
    version='0.2',
    description='Transcribe (digitize) handwritten manuscripts and digitizer pen input',
    author='Daniel Vorberg',
    author_email='dv@pks.mpg.de',
    url='https://github.com/antemons/smart-manuscript',
    packages=['smart_manuscript'],
    package_dir={'smart_manuscript': 'smart_manuscript'},
    scripts=["transcribe", "manuscript-writer"],
    include_package_data = True,
    long_description=""" """)
