from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()


setup(
    name='meso_fab_mc',
    version='0.1.0',    
    description="Mesoscopic Monte-Carlo Fabric Evolution Model",
    url='https://github.com/danrichards678/',
    author='Daniel Richards',
    author_email='danrichards678@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['meso_fab_mc'],
    install_requires=['numpy',                     
                      ],
    extras_require = {
        'harmonics': ['shtns','scipy']
                    },

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
)
