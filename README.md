# Tomato Analyzer Lite

Extract geometric trait measurements from images of tomatoes and other fruit.

Clustering and segmentation by [Suxing Liu](https://github.com/lsx1980/plant-image-analysis). Trait extraction adapted by [Wes Bonelli](mailto:wbonelli@uga.edu).

## Requirements

Either [Docker](https://www.docker.com/) or [Singularity](https://sylabs.io/singularity/) is required to run this project in a Unix environment.

## Installation

Clone the project with `git clone https://github.com/van-der-knaap-lab/tomato-analyzer-lite.git`.

## Usage

To analyze files in a directory relative to the project root:

#### Docker

```bash
docker run -v "$(pwd)":/opt/tomato-analyzer-lite -w /opt/tomato-analyzer-lite wbonelli/tomato-analyzer-lite python3.8 /opt/tomato-analyzer-lite/talite.py -i inputfile -o output/directory
```

#### Singularity

```bash
singularity exec docker://wbonelli/tomato-analyzer-lite python3.8 talite.py -i inputfile -o output/directory
```
