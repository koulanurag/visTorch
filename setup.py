from os import path
from setuptools import setup

setup(name='visTorch',
      version='0.0.1',
      py_modules=['visTorch'],
      packages=['visTorch'],
      author='Anurag Koul',
      author_email='koulanurag@gmail.com',
      long_description=open(path.join(path.abspath(path.dirname(__file__)), 'README.md')).read(),
      license='MIT')
