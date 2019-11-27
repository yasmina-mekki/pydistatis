import setuptools


setuptools.setup(
	name='pydistatis',
	version='0.0.0a',
	packages=['pydistatis'],
	author='yasmina-mekki',
	author_email='yas.mekki@gmail.com',
	description='set of functions to use distatis',
	long_description=open('README.md').read(),
	url='https://github.com/yasmina-mekki/pydistatis',
	install_requires=['numpy', 'pandas'],
	python_requires='>=3.6'
	)
