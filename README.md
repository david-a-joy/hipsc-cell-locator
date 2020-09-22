# Human Annotator Interface for Tagging Cell Nuclei

Blinded and randomized presentation of cell nuclei images for human annotation

If you find this code useful, please cite:

> Joy, D. A., Libby, A. R. G. & McDevitt, T. C. Deep neural net tracking of
> human pluripotent stem cells reveals intrinsic behaviors directing morphogenesis.
> https://www.biorxiv.org/content/10.1101/2020.09.21.307470v1 (2020) doi:10.1101/2020.09.21.307470.

## Installing

This script requires Python 3.7 or greater and several additional python packages.
This code has been tested on OS X 10.15 and Ubuntu 18.04, but may work with minor
modification on other systems.

It is recommended to install and test the code in a virtual environment for
maximum reproducibility:

```{bash}
# Create the virtual environment
python3 -m venv ~/org_env
source ~/org_env/bin/activate
```

All commands below assume that `python3` and `pip3` refer to the binaries installed in
the virtual environment. Commands are executed from the base of the git repository
unless otherwise specified.

```{bash}
pip3 install --upgrade pip

# Install the required packages
pip3 install -r requirements.txt
```

After installation, the the cell locator can be run with the following command:

```{bash}
python3 cell_locator.py
```

Annotators should attempt to complete the annotation task in one sitting for comparability
with the results in (Joy et al, 2020).

After completion, annotators should send the `RegionDB.sqlite3` database to the
researcher for analysis using the [deep-hipsc-tracking](https://github.com/david-a-joy/deep-hipsc-tracking) toolbox.
