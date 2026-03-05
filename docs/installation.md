## 1. Installation

### 1.1 System requirements

Make sure [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/), [git-lfs](https://git-lfs.com/) are installed.

Depending on your system, it should look like:
```shell
# For downloading model weights
sudo apt install git-lfs       # ubuntu
brew install git-lfs           # macOS
git-lfs install                # both

# For audio processing
sudo apt install ffmpeg         # ubuntu
brew install ffmpeg             # macOS

# For managing python packages
curl -LsSf https://astral.sh/uv/install.sh | sh # both
```

Note that BabAR has been tested on Linux and macOS only. Windows is untested and may require additional setup.

### 1.2 Install BabAR

You can then run the following commands:

```sh
# Clone repository
git clone --recurse-submodules https://github.com/MarvinLvn/BabAR.git

# Install python dependencies
cd BabAR
uv sync
```
