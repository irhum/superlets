# Superlets

This repository provides an unofficial JAX/Python implementation of the Superlets algorithm introduced in [Time-frequency super-resolution with superlets](https://www.nature.com/articles/s41467-020-20539-9). The original MATLAB implementation can be found [here](https://github.com/TransylvanianInstituteOfNeuroscience/Superlets).

### Demo
In time-frequency analysis, we convert a signal into a time-frequency representation, where we want to identify how "strongly" each frequency component is activated at each timestep. Due to the [Gabor limit](https://en.wikipedia.org/wiki/Uncertainty_principle#Signal_processing), there are fundamental limitations to how precise our estimates can be: higher frequency precision requires trading off precision in time, and vice versa. In wavelet analysis, this is typically controlled by the number of cycles used in each wavelet. 

For instance, if we have the following signal (similar to that of Fig. 3 in the paper):

[Signal](../images/signal.png?raw=true)

We can obtain three different scalograms by varying the number of cycles used per wavelet. More cycles result in a more precise estimate of frequency, at the expense of precision in time:

[Wavelet](../images/wavelet.png?raw=true)

However, the proposed solution in the superlets paper is that Wavelet transforms aren't Pareto optimal (that is, you can improve on *both* time and frequency without having to make trade-offs), and it can be improved on by taking the geometric mean of wavelets of different cycles, resulting in the following:

[Wavelet](../images/superlet.png?raw=true)

#### Note
This algorithm uses a non-standard normalization for the wavelets used; typically, the square of the wavelets are normalized to a value of 1, here the absolute value is. The authors provide a justification for this in the [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-20539-9/MediaObjects/41467_2020_20539_MOESM1_ESM.pdf) on page 7. Both figures above here use absolute value normalization.
