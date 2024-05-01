## LEO-vetter: for Lazy Exoplanet Operations ##

LEO-vetter is a tool for automated vetting of transit signals found in light curve data. Inspired by the Kepler Robovetter, LEO computes vetting metrics and then checks those metrics against a series of pass-fail thresholds. If a signal passes all checks, it is considered a planet candidate (PC). If a signal fails at least one test, it may be either an astrophysical false positive (FP; e.g. eclipsing binary, nearby eclipsing signal) or false alarm (FA; e.g. systematic, stellar variability). LEO also produces vetting reports for quick manual inspection of the signal.

Flux-level vetting can work for Kepler, K2, and TESS data. Pixel-level vetting has been implemented for TESS usage only.

### Usage ###

Check out the tutorials for full usage, but at its simplest it will look something like the following:

```
from leo_vetter.main import TCELightCurve
from leo_vetter.pixel import pixel_vetting
from leo_vetter.thresholds import check_thresholds

tlc = TCELightCurve(ID, time, raw, flux, flux_err, period, epoch, duration)

# Run flux-level vetting
# "star" is a dict containing stellar properties like mass, radius, etc.
tlc.compute_flux_metrics(star)

# Run pixel-level vetting
# "sectors" is a list of desired sectors for making difference images
tdi, good_sectors, good_pixel_data, good_centroids = pixel_vetting(tlc, star, sectors)

# Check metrics against pass-fail thresholds
FA = check_thresholds(tlc.metrics, "FA")
FP = check_thresholds(tlc.metrics, "FP")
```

Important note: The thresholds that determine whether a signal passes or fails are still undergoing optimization (and no single set of thresholds should work for all use-cases!), but the current thresholds should work pretty well for TESS-observed FGKM dwarf stars.

### Installation ###

```
git clone https://github.com/mkunimoto/LEO-vetter.git
cd LEO-vetter
pip install .
```

If you also want to run pixel-level vetting (recommended), you will need to install the `transit-diffImage` package available [here](https://github.com/stevepur/transit-diffImage):

```
git clone https://github.com/stevepur/transit-diffImage.git
cd transit-diffImage
pip install .
```

### Citing LEO ###

There is a paper upcoming to describe LEO, but for now please cite this codebase here if you found it useful: [https://ascl.net/2404.026](https://ascl.net/2404.026).
