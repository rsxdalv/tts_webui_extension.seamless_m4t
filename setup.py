import setuptools
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setuptools.setup(
    name="tts_webui_extension.seamless_m4t",
    packages=setuptools.find_namespace_packages(),
    version="0.0.7",
    author="rsxdalv",
    description="SeamlessM4T is a multilingual and multimodal translation model supporting text and speech",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rsxdalv/tts_webui_extension.seamless_m4t",
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

