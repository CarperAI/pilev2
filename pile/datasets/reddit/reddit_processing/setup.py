import setuptools
setuptools.setup(
   name='dataflow_dependencies',
   version='1.0',
   install_requires=[
        "ujson",
        "jsonlines",
   ],
   packages=setuptools.find_packages()
)