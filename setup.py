from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()


setup(
    name='mcfab',
    version='0.1.0',    
    description="Monte-Carlo Fabric Evolution Model",
    url='https://github.com/dhrichards/',
    author='Daniel Richards',
    author_email='danrichards678@gmail.com',
    license='GPL-3.0',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['mcfab'],
    package_data={'mcfab':['data/*']},
    install_requires=['numpy','jax','jaxlib','jaxopt'                  
                      ],
    extras_require = {
        'harmonics': ['shtns','scipy']
                    },

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL-3.0 License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
)
