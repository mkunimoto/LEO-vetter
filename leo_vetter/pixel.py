import os
import numpy as np
import pickle
import csv

from transitDiffImage import tessDiffImage, transitCentroids, tessprfmodel
from tess_stars2px import tess_stars2px_function_entry as ts2px


def star_dict(tic, ra, dec, quality_flags=None):
    star_data = {}
    star_data["id"] = tic
    star_data["raDegrees"] = ra
    star_data["decDegrees"] = dec
    star_data["qualityFiles"] = None
    star_data["qualityFlags"] = quality_flags
    return star_data


def planet_dict(tic, planetno, per, epo, dur):
    planet_data = {}
    planet_data["planetID"] = f"{tic}.{planetno}"
    planet_data["period"] = per
    planet_data["epoch"] = epo
    planet_data["durationHours"] = dur * 24
    return planet_data


def planet_dict_from_metrics(metrics):
    tic = int(metrics["tic"])
    planetno = int(metrics["planetno"])
    if ~np.isnan(metrics["transit_aic"]):
        per = metrics["transit_per"]
        epo = metrics["transit_epo"]
        dur = metrics["transit_dur"]
    elif ~np.isnan(metrics["trap_aic"]):
        per = metrics["trap_per"]
        epo = metrics["trap_epo"]
        dur = metrics["trap_dur"]
    else:
        per = metrics["per"]
        epo = metrics["epo"]
        dur = metrics["dur"]
    return planet_dict(tic, planetno, per, epo, dur)


def prf_fit(sector, cam, ccd, images, catalogue):
    prf = tessprfmodel.SimpleTessPRF(
        shape=images["diffImage"].shape,
        sector=sector,
        camera=cam,
        ccd=ccd,
        column=catalogue["extent"][0],
        row=catalogue["extent"][2],
    )
    fit_vector, quality, _, _, _ = transitCentroids.tess_PRF_centroid(
        prf, catalogue["extent"], images["diffImage"], catalogue
    )
    # Find the offset in terms of pixels
    offset_col = np.abs(fit_vector[0] - catalogue["ticColPix"][0])
    offset_row = np.abs(fit_vector[1] - catalogue["ticRowPix"][0])
    offset_pix = np.sqrt(offset_col**2 + offset_row**2)
    # Find the offset in terms of arcsecs
    ra_dec = tessDiffImage.pix_to_ra_dec(sector, cam, ccd, fit_vector[0], fit_vector[1])
    corrected_ra = catalogue["correctedRa"][0]
    corrected_dec = catalogue["correctedDec"][0]
    offset_ra = ra_dec[0] - corrected_ra
    offset_dec = ra_dec[1] - corrected_dec
    offset_arc = 3600 * np.sqrt(
        (offset_ra * np.cos(corrected_dec * np.pi / 180)) ** 2 + offset_dec**2
    )
    # Save all results
    results = {}
    results["sector"] = sector
    results["cam"] = cam
    results["ccd"] = ccd
    results["quality"] = quality
    results["offset_pix"] = offset_pix
    results["offset_arc"] = offset_arc
    results["tic_col"] = catalogue["ticColPix"][0]
    results["tic_row"] = catalogue["ticRowPix"][0]
    results["fit_col"] = fit_vector[0]
    results["fit_row"] = fit_vector[1]
    return results


def multisector_images(
    star, sectors, planet_idx=0, save_dir=".", n_bad=0, max_sectors=np.inf
):
    tic = star["id"]
    planet = star["planetData"][planet_idx]
    planet_ID = planet["planetID"]
    _, _, _, all_sectors, all_cams, all_ccds, _, _, _ = ts2px(
        tic, star["raDegrees"], star["decDegrees"]
    )
    good_sectors = []
    good_pixel_data = []
    good_centroids = []
    for sector in sectors:
        star["sector"] = sector
        star["cam"] = all_cams[all_sectors == sector][0]
        star["ccd"] = all_ccds[all_sectors == sector][0]
        tdi = tessDiffImage.tessDiffImage(star, outputDir=save_dir)
        pixel_file = os.path.join(
            save_dir, f"tic{tic}/imageData_{planet_ID}_sector{sector}.pickle"
        )
        try:
            tdi.make_ffi_difference_image(
                thisPlanet=planet_idx, allowedBadCadences=n_bad
            )
        except (KeyError, ValueError, FileNotFoundError):
            continue
        if not os.path.exists(pixel_file):
            print(f"TIC-{planet_ID}: difference image failed for sector {sector}")
            continue
        try:
            with open(pixel_file, "rb") as f:
                pixel_data = pickle.load(f)
        except EOFError:
            print(f"TIC-{planet_ID}: difference image failed for sector {sector}")
            _ = os.system(f"rm {pixel_file}")
            continue
        images = pixel_data[0]
        catalogue = pixel_data[1]
        if "diffImage" not in images:
            print(f"TIC-{planet_ID}: difference image failed for sector {sector}")
        elif np.isnan(np.sum(images["diffImage"])):
            print(f"TIC-{planet_ID}: difference image failed for sector {sector}")
        else:
            centroid = prf_fit(sector, star["cam"], star["ccd"], images, catalogue)
            good_sectors.append(sector)
            good_pixel_data.append(pixel_data)
            good_centroids.append(centroid)
        if len(good_sectors) == max_sectors:
            print(
                f"TIC-{planet_ID}: reached max sector limit ({max_sectors}); stopping"
            )
            break
    return tdi, good_sectors, good_pixel_data, good_centroids


def pixel_vetting(
    tlc, star, sectors, quality_flags=None, tdi_dir=".", n_bad=0, max_sectors=np.inf
):
    star_data = star_dict(
        star["tic"], star["ra"], star["dec"], quality_flags=quality_flags
    )
    planet_data = planet_dict_from_metrics(tlc.metrics)
    star_data["planetData"] = [planet_data]
    tdi, good_sectors, good_pixel_data, good_centroids = multisector_images(
        star_data, sectors, save_dir=tdi_dir, n_bad=n_bad, max_sectors=max_sectors
    )
    # offset_mean = mean offset weighted by quality
    # offset_qual = offset with best quality
    if len(good_centroids) == 0:
        tlc.metrics["offset_mean"] = np.nan
        tlc.metrics["offset_qual"] = np.nan
    else:
        offset_arc = np.array([centroid["offset_arc"] for centroid in good_centroids])
        quality = np.array([centroid["quality"] for centroid in good_centroids])
        tlc.metrics["offset_mean"] = np.nanmean(offset_arc * quality) / np.nansum(
            quality
        )
        tlc.metrics["offset_qual"] = offset_arc[np.argmax(quality)]
    return tdi, good_sectors, good_pixel_data, good_centroids
