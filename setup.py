import setuptools

setuptools.setup(
    name="extension_seamless_m4t",
    packages=setuptools.find_namespace_packages(),
    version="0.0.2",
    author="rsxdalv",
    description="SeamlessM4T is a multilingual and multimodal translation model supporting text and speech",
    url="https://github.com/rsxdalv/extension_seamless_m4t",
    project_urls={},
    scripts=[],
    install_requires=[
        "transformers>=4.30.0",
        "torchaudio>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
