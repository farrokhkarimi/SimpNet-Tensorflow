from setuptools import setup

def get_requirements():
      reqs = open('requirements.txt').read()
      return [req.split() for req in reqs if req != ""]

setup(name='simpnet',
      version='0.1',
      description='A Tensorflow Implementation of the SimpNet Convolutional Neural Network Architecture',
      url='http://github.com/hexpheus/SimpNet-Tensorflow',
      author='Ali Gholami',
      author_email='hexpheus@gmail.com',
      install_requires=get_requirements(),
      license='MIT',
      packages=['simpnet'],
      zip_safe=False)