## 1. Installation

### 1.1 System requirements

Make sure [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/), [git-lfs](https://git-lfs.com/) are installed.

Depending on your system, it should look like:
```shell
# For downloading model weights
sudo apt install git-lfs       # Ubuntu/Debian
brew install git-lfs            # macOS
git-lfs install

# For audio processing
sudo apt install ffmpeg         # Ubuntu/Debian
brew install ffmpeg             # macOS

# For managing python packages
curl -LsSf https://astral.sh/uv/install.sh | sh # Ubuntu/Debian/macOS
```

Note that BabAR has been tested on Linux and macOS only. Windows is untested and may require additional setup.

### 1.2 Install BabAR

You can then run the following commands:

```sh
# Clone repository
git clone --recurse-submodules https://github.com/MarvinLvn/BabAR.git

# Get VTC2.0 and BabAR's weights
cd BabAR
git submodule foreach --recursive git lfs pull

# Install python dependencies
uv sync
```