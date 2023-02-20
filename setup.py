from setuptools import setup, find_packages

setup(name='multiagent',
      version='0.0.1',
      description='Multi-Agent Goal-Driven Communication Environment',
      url='https://github.com/openai/multiagent-public',
      author='Igor Mordatch',
      author_email='mordatch@openai.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
            'gym==0.20.0',
            'numpy',
            'numpy-stl',
            'pygame'
      ]
)

#add six==1.15.0
#add pyglet=1.5.11