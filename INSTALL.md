# Installation Instructions for LitArt

### All systems (Linux, and Windows)

Important: do not use your OS package manager (like apt-get) nor pip to install Python and its main dependencies. Use Anaconda or Miniconda instead.


Download the version according to your machine
1. <a href="https://github.com/conda-forge/miniforge">mini-forge-conda</a>  
2. <a href="https://www.python.org/downloads/">Python 3.9</a>

### Cloning the Repository
``` git clone https://github.com/ronair212/LitArt.git ```

### Create Environment
``` conda create -n <myenv> python=3.9 ```

### Troubleshoot

If you face ``` segmentation fault``` as an error while generating images do as follows: <br>
1. ``` module load anaconda3/2022.05 ```
2. ``` module load cuda/12.1 ```

Load the version that is already installed on your machine


