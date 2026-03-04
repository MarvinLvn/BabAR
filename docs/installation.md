## 1. Installation

### 1.1 System requirements

Make sure the following tools are installed on your system:
- [uv](https://docs.astral.sh/uv/) a python package manager
- [ffmpeg](https://ffmpeg.org/) used for audio processing
- [git-lfs](https://git-lfs.com/) required to download model weights

BabAR has been tested on Linux and macOS only. Windows is untested and may require additional setup

### 1.2 Install BabAR

You can then run the following commands:

```sh
# Clone repository
git-lfs install
git clone --recurse-submodules https://github.com/MarvinLvn/BabAR.git

# Get VTC2.0 and BabAR's weights
cd BabAR
git submodule foreach --recursive git lfs pull

# Install python dependencies
uv sync
```