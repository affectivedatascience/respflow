# RespFlow

A Python package for preprocessing and feature extraction of respiration signals for Respiratory Inductance Plethysmography (RIP) belts.

![Dashboard Demo](./readme_assets/dashboard_demo.gif)

## Statement Of Need



## Example

```python
import RespFlow as rf

# Get path dictionary
path_names = rf.make_paths('./data')

# Load sample data
rf.make_sample_data(path_names)

# Preprocess signals
rf.clean_signals(path_names, sampling_rate=2000)

# Plot data on the "Respiration" column
rf.plot_dashboard(path_names, 'Respiration')
```
## Installation

The package has not yet been deployed to PyPi, but feel free to git clone the repo and install it locally!

```bash 
pip install -e
```

You can then import it with

```python
import RespFlow as rf
```


