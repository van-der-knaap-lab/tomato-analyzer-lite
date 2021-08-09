# Tomato Analyzer Lite

Extract geometric trait measurements from images of tomatoes and other fruit.

## Acknowledgements

General approach inspired by Suxing Liu, in particular [Smart Plant Growth Top-Down Traits](https://github.com/Computational-Plant-Science/spg).

[PlantCV](https://github.com/danforthcenter/plantcv) also used for skeletonization, pruning, leaf & stem counting, etc.

## Requirements

[Docker](https://www.docker.com/) is required to run this project in a Unix environment.

## Installation

To install from source, clone the project with `git clone https://github.com/van-der-knaap-lab/tomato-analyzer-lite.git`, then build the image from the root directory with `docker build -t <your tag> -f Dockerfile .`.

Alternatively, you can just pull the pre-built image with `docker pull van-der-knaap-lab/tomato-analyzer-lite`, or allow it to be pulled automatically from another Docker CLI command (as below).

## Usage

To analyze an image:

```bash
docker run wbonelli/tomato-analyzer-lite python3.8 /opt/tomato-analyzer-lite/talite.py <input file>
```

By default, output files will be written to the current working directory. To specify a different output location, use `-o <full path to output directory>`.

## Development

To set up a development environment and explore or modify the source, just mount the project root as your container's working directory, for instance:

```bash
docker run -it -v $(pwd):/opt/dev -w /opt/dev wbonelli/tomato-analyzer-lite bash
```

Then invoke the CLI with `python3.8 /opt/code/cli.py <input file>`.

Git is also instructed to ignore `data` and `output` directories for convenience.
