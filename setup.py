from setuptools import setup, find_packages


setup(name='acse',
      version='1.0.0',
      description='',
      author='',
      author_email='',
      # url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu'],
      packages=find_packages(),
    )
