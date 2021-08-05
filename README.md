# Superlets

This repository provides a JAX/Python implementation of the Superlets algorithm introduced in [Time-frequency super-resolution with superlets](https://www.nature.com/articles/s41467-020-20539-9). The original MATLAB implementation can be found [here](https://github.com/TransylvanianInstituteOfNeuroscience/Superlets).

### Demo
In time-frequency analysis, we convert a signal into a time-frequency representation, where we want to identify how "strongly" each frequency component is activated at each timestep. Due to the [Gabor limit](https://en.wikipedia.org/wiki/Uncertainty_principle#Signal_processing), there are fundamental limitations to how precise our estimates can be: higher frequency precision requires trading off precision in time, and vice versa. In wavelet analysis, this is typically controlled by the number of cycles used in each wavelet. 

For instance, if we have the following signal:

We can obtain three different scalograms by varying the number of cycles used per wavelet. More cycles result in a more precise estimate of frequency, at the expense of precision in time:

[]()

Superlets algorithm for time-frequency analysis implemented in JAX/Python
