import os, sys, pathlib
import argparse
from tqdm import tqdm

import uproot
import numpy as np
import pandas as pd
import awkward as ak
import scipy
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)
plt.rcParams.update({'font.size': 32})

hep.cms.label()#, data=<True|False>, lumi=50, year=2017)

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(8)

chamber_text = "Tracker {}\n"\
        +"10x10 $cm^2$ triple-GEM\n"\
        +"250 µm strip pitch\n"\
        +"Ar-$CO_2$ 70%-30%\n"\
        +"$5\,M#Omega$ high voltage divider\n"\

def linear_function(x, *p):
    q, m = p
    return q + m*x

def gauss(x, *p):
    A, mu, sigma = p
    return A*scipy.stats.norm.pdf(x, loc=mu, scale=sigma)
    #return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gauss2(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    # to be checked
    return A1*scipy.stats.norm.pdf(x, loc=mu1, scale=sigma1) + A2*scipy.stats.norm.pdf(x, loc=mu2, scale=sigma2)
    #return gauss(x, A1, mu1, sigma1) + gauss(x, A2, mu2, sigma2)

def analyse_residuals(residuals, histo_range, nbins, ax, legend, xlabel, color="red"):
   
    #histo_range = (residuals.min(), residuals.max())
    points, bins = np.histogram(residuals, bins=nbins, range=histo_range)
    bins = bins[:-1]+ 0.5*(bins[1:] - bins[:-1])
    
    # gaussian fit
    coeff = [len(residuals), ak.mean(residuals), ak.std(residuals)]
    perr = [0]*len(coeff)
    #coeff += [len(residuals)*0.1, ak.mean(residuals), 10*ak.std(residuals)]
    #print(gauss2, "bins:", bins, "\npoints:", points, coeff)
    try:
        coeff, var_matrix = curve_fit(gauss, bins, points, p0=coeff, method="lm")
        perr = np.sqrt(np.diag(var_matrix))
    except RuntimeError:
        print("Fit failed, using RMS instead...")
    
    correction, err_correction = coeff[1], perr[1]
    space_resolution, err_space_resolution = 1e3*coeff[2], 1e3*perr[2]
    
    # plot data and fit
    ax.hist(
        residuals, bins=nbins, range=histo_range,
        histtype="stepfilled", linewidth=2, facecolor="none", edgecolor="k",
        label = legend
    )
    #ax.scatter(bins, points, marker="o", label=label)
    xvalues = np.linspace(bins[0], bins[-1], 1000)
    ax.plot(xvalues, gauss(xvalues, *coeff), color=color, linewidth=2)
    ax.set_xlabel(xlabel)
    binning_um = 1e3*(bins[1]-bins[0])
    ax.set_ylabel(f"Events/{binning_um:1.0f} µm")

    return correction, err_correction, space_resolution, err_space_resolution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ifile", type=pathlib.Path, help="Input file")
    parser.add_argument('odir', type=pathlib.Path, help="Output directory")
    parser.add_argument("-n", "--events", type=int, default=-1, help="Number of events to analyse")
    parser.add_argument("--by-cluster-size", action="store_true", help="Plot profile as 2D plot with cluster size")
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate logging")
    args = parser.parse_args()
    
    os.makedirs(args.odir, exist_ok=True)

    with uproot.open(args.ifile) as track_file:
        track_tree = track_file["trackTree"]
        if args.verbose: track_tree.show()

        print("Reading tree...")
        rechits_chamber = track_tree["rechits2D_Chamber"].array(entry_stop=args.events)
        rechits_x = track_tree["rechits2D_X"].array(entry_stop=args.events)
        rechits_y = track_tree["rechits2D_Y"].array(entry_stop=args.events)
        prophits_x = track_tree["prophits2D_X"].array(entry_stop=args.events)
        prophits_y = track_tree["prophits2D_Y"].array(entry_stop=args.events)
        
        #rechits_x_error = track_tree["rechits2D_X_Error"].array(entry_stop=args.events)
        #rechits_y_error = track_tree["rechits2D_Y_Error"].array(entry_stop=args.events)
        track_chi2_x = track_tree["tracks_X_chi2"].array(entry_stop=args.events)
        track_chi2_y = track_tree["tracks_Y_chi2"].array(entry_stop=args.events)
        tracks_x_covariance = track_tree["tracks_X_covariance"].array(entry_stop=args.events)
        tracks_y_covariance = track_tree["tracks_Y_covariance"].array(entry_stop=args.events)
        cluster_size_x = track_tree["rechits2D_X_ClusterSize"].array(entry_stop=args.events)
        cluster_size_y = track_tree["rechits2D_Y_ClusterSize"].array(entry_stop=args.events)
        prophits_x_error = track_tree["prophits2D_X_Error"].array(entry_stop=args.events)
        prophits_y_error = track_tree["prophits2D_Y_Error"].array(entry_stop=args.events)
       
        n_chambers = 3
        # choose only events with hits in all chambers:
        mask_nhits = ak.count_nonzero(rechits_chamber>=0, axis=1)>=n_chambers
        rechits_chamber = rechits_chamber[mask_nhits]
        rechits_x, rechits_y = rechits_x[mask_nhits], rechits_y[mask_nhits]
        prophits_x, prophits_y = prophits_x[mask_nhits], prophits_y[mask_nhits]
        cluster_size_x, cluster_size_y = cluster_size_x[mask_nhits], cluster_size_y[mask_nhits]
        prophits_x_error, prophits_y_error = prophits_x_error[mask_nhits], prophits_y_error[mask_nhits]
        tracks_x_covariance, tracks_y_covariance = tracks_x_covariance[mask_nhits], tracks_y_covariance[mask_nhits]
        track_chi2_x, track_chi2_y = track_chi2_x[mask_nhits], track_chi2_y[mask_nhits]
 
        if args.verbose:
            print("Rechits x:", rechits_x, "length:", ak.count(rechits_x, axis=1))
            print("Prophits y:", prophits_x, "length:", ak.count(prophits_x, axis=1))
        residuals_x = prophits_x - rechits_x
        residuals_y = prophits_y - rechits_y
       
        # Preparing figures:
        print("Starting plotting...")
        directions = ["x", "y"]
        residual_fig, residual_axs = plt.subplots(nrows=2, ncols=n_chambers, figsize=(12*n_chambers,20))
        residual_cls_fig, residual_cls_axs = plt.subplots(nrows=2, ncols=n_chambers, figsize=(12*n_chambers,18))
        spres_fig, spres_axs = plt.subplots(nrows=1, ncols=n_chambers, figsize=(12*n_chambers,10))
        rotation_fig, rotation_axs = plt.subplots(nrows=2, ncols=n_chambers, figsize=(12*n_chambers,18))
        properr_fig, properr_axs = plt.subplots(nrows=1, ncols=n_chambers, figsize=(12*n_chambers,7))
        prophits_fig, prophits_axs = plt.subplots(nrows=2, ncols=n_chambers, figsize=(12*n_chambers,18))
        profile_fig, profile_axs = plt.subplots(nrows=2, ncols=n_chambers, figsize=(12*n_chambers,18)) 
        residuals2d_xx_fig, residuals2d_xx_axs = plt.subplots(nrows=2, ncols=n_chambers, figsize=(12*n_chambers,18))
        residuals2d_xy_fig, residuals2d_xy_axs = plt.subplots(nrows=2, ncols=n_chambers, figsize=(12*n_chambers,18))
        cluster_size_fig, cluster_size_axs = plt.subplots(nrows=1, ncols=n_chambers, figsize=(12*n_chambers,9))
        chi2_fig, chi2_axs = plt.subplots(nrows=2, ncols=n_chambers, figsize=(10*n_chambers,9*2))
        properr_position_fig, properr_position_axs = plt.subplots(nrows=2, ncols=n_chambers, figsize=(10*n_chambers,9*2))

        angles, err_angles = np.ndarray((2,n_chambers)), np.ndarray((2,n_chambers))
        translation, err_translation = np.ndarray((2,n_chambers)), np.ndarray((2,n_chambers))
        for tested_chamber in tqdm(range(n_chambers)):
            rechits = [rechits_x[:,tested_chamber], rechits_y[:,tested_chamber]]
            # apply angular correction to rechits:
            # rechits_corrected = [
            #     rechits[0]*np.cos(ANGLES[tested_chamber]) + rechits[1]*np.sin(ANGLES[tested_chamber]),
            #     -rechits[0]*np.sin(ANGLES[tested_chamber]) + rechits[1]*np.cos(ANGLES[tested_chamber])
            # ]
            # rechits = rechits_corrected

            prophits = [np.array(prophits_x[:,tested_chamber]), np.array(prophits_y[:,tested_chamber])]
            residuals = [np.array(prophits[0]-rechits[0]), np.array(prophits[1]-rechits[1])]
            cluster_sizes = [np.array(cluster_size_x[:,tested_chamber]), np.array(cluster_size_y[:,tested_chamber])]
            tracks_covariance = [np.array(tracks_x_covariance[:,tested_chamber]), np.array(tracks_y_covariance[:,tested_chamber])]
            properrs = np.array(prophits_x_error[:,tested_chamber]), np.array(prophits_y_error[:,tested_chamber])
            chi2 = [np.array(track_chi2_x[:,tested_chamber]), np.array(track_chi2_y[:,tested_chamber])]

            space_resolutions, err_space_resolutions = dict(), dict()
            cluster_size_cuts = list(range(2,10))

            for idirection in range(2):
                direction = directions[idirection]
                space_resolutions[direction], err_space_resolutions[direction] = list(), list()
                cluster_size = cluster_sizes[idirection]

                if args.by_cluster_size:
                    """profile_axs[idirection][tested_chamber].hist2d(
                        ak.to_numpy(rechits[idirection]),
                        cluster_size, bins=(100,10),
                        range=((ak.min(rechits[idirection]),ak.max(rechits[idirection])), (0, 10))
                    )"""
                    profile_axs[idirection][tested_chamber].plot(
                        ak.to_numpy(rechits[idirection]),
                        cluster_size, "."
                    )

                else:
                    profile_axs[idirection][tested_chamber].hist(rechits[idirection], bins=100)

                profile_axs[idirection][tested_chamber].set_xlabel(f"Reconstructed {direction} (mm)")
                profile_axs[idirection][tested_chamber].set_ylabel("Cluster size")
                profile_axs[idirection][tested_chamber].set_title(f"Tracker {tested_chamber}")

                # plot propagated hits
                prophits_axs[idirection][tested_chamber].hist(
                    prophits[idirection], bins=200, range=(-0.5,0.5),
                    histtype="stepfilled", linewidth=1, facecolor="none", edgecolor="k",
                )
                prophits_axs[idirection][tested_chamber].set_title(
                    f"BARI-0{tested_chamber+1} {direction}"
                )
                binsize = 1/200*1e3
                prophits_axs[idirection][tested_chamber].set_xlabel("Propagated hit position (mm)")
                prophits_axs[idirection][tested_chamber].set_ylabel(f"Events/{int(binsize)} µm")

                # plot residuals for all cluster sizes:
                correction, err_correction, space_resolution, err_space_resolution = analyse_residuals(
                    residuals[idirection],
                    (-4.7, 4.7), 200,
                    residual_axs[idirection][tested_chamber],
                    "", f"Residual {direction} (mm)",
                    color=["red", "blue"][idirection]
                )
                residual_axs[idirection][tested_chamber].text(
                    0.05, 0.9,
                    chamber_text.format(tested_chamber+1),
                    transform=residual_axs[idirection][tested_chamber].transAxes,
                    va="top", linespacing=1.7
                )
                residual_axs[idirection][tested_chamber].text(
                    0.95, 0.9, 
                    f"$\sigma$ = {space_resolution:1.1f} $\pm$ {err_space_resolution:1.1f} µm",
                    transform=residual_axs[idirection][tested_chamber].transAxes,
                    va="top", ha="right"
                )
                hep.cms.text(text="Preliminary", ax=residual_axs[idirection][tested_chamber])

                translation[idirection][tested_chamber] = correction
                err_translation[idirection][tested_chamber] = err_correction
 
                chi2_axs[idirection][tested_chamber].hist2d(
                    chi2[idirection], 1e3*tracks_covariance[idirection],
                    bins=50, range=((0,1), (0,0.1)),
                )
                #chi2_axs[idirection][tested_chamber].hist(1e3*tracks_covariance[idirection], bins=50)
                #chi2_axs[idirection][tested_chamber].set_yscale("log")
                chi2_axs[idirection][tested_chamber].set_xlabel(f"χ$^2_{direction}$")
                chi2_axs[idirection][tested_chamber].set_ylabel("Fit parameters covariance (µm)")
                
                residual_cls_axs[idirection][tested_chamber].set_title(f"BARI-0{tested_chamber+1} {direction}")
                # plot residuals for cluster sizes separately:
                for cls in cluster_size_cuts:
                    corr, err_corr, res, err_res = analyse_residuals(
                        residuals[idirection][cluster_size==cls],
                        (-3.45, 3.45), 120,
                        residual_cls_axs[idirection][tested_chamber],
                        f"cluster size {cls}",
                        f"{directions[idirection]} residual (mm)"
                    )
                    space_resolutions[direction].append(res)
                    err_space_resolutions[direction].append(err_res)

                    # residual_cls_axs[idirection][tested_chamber].text(
                    #     2, (1-0.1*parity)*1e6,
                    #     f"Space resolution {space_resolution:1.0f} µm",
                    #     horizontalalignment="right",
                    #     fontsize=20
                    # )
                
                # plot residuals vs propagated position:
                prophit_bins = np.linspace(-30, 30, 15)
                prophit_means, residual_means = list(), list()
                prophit_errors, residual_errors = list(), list()
                for i,b in enumerate(prophit_bins[:-1]):
                    b_min, b_max = b, prophit_bins[i+1]
                    selection = (prophits[idirection]>b_min) & (prophits[idirection]<b_max)
                    prophit_means.append(ak.mean(prophits[idirection][selection]))
                    residual_means.append(ak.mean(residuals[idirection-1][selection]))
                    prophit_errors.append(ak.std(prophits[idirection][selection]))
                    residual_errors.append(ak.std(residuals[idirection-1][selection])/np.sqrt(ak.num(residuals[idirection][selection], axis=0)))
                   
                """ determine rotation corrections: """
                rotation_axs[idirection][tested_chamber].errorbar(
                    prophit_means, residual_means, xerr=prophit_errors, yerr=residual_errors, fmt="o"
                )
                # fit with line and plot result:
                coeff = [
                    0.5*(residual_means[0]+residual_means[-1]),
                    (residual_means[-1]+residual_means[0])/(prophit_means[-1]+prophit_means[0])
                ]
                try:
                    coeff, var_matrix = curve_fit(linear_function, prophit_means, residual_means, p0=coeff, method="lm")
                    q, m = coeff
                    err_q, err_m = np.sqrt(np.diag(var_matrix))
                except Exception as e: print("Skipping rotation fit:", e)

                # calculate angles:
                theta = np.arcsin(m)
                err_theta = err_m/np.sqrt(1-m**2)
                angles[idirection][tested_chamber] = (-1)**idirection*theta
                err_angles[idirection][tested_chamber] = err_theta

                x_fit = np.linspace(-30, 30, 20)
                rotation_axs[idirection][tested_chamber].plot(x_fit, linear_function(x_fit, *coeff), color="red")
                rotation_axs[idirection][tested_chamber].text(
                    0.7, 0.8,
                    f"m = {m:1.1e} $\pm$ {err_m:1.1f}\n"+
                    f"q = {q*1e3:1.2f} $\pm$ {err_q*1e3:1.1f} µm\n"+
                    f"ϑ = {theta*1e3:1.1f} $\pm$ {err_theta*1e3:1.1f} mrad",
                    transform=rotation_axs[idirection][tested_chamber].transAxes,
                    bbox=dict(boxstyle="square, pad=0.5", ec="black", fc="none")
                )

                #rotation_axs[idirection][tested_chamber].set_xlim(-30, 30)
                #rotation_axs[idirection][tested_chamber].set_ylim(-0.06, 0.06)
                rotation_axs[idirection][tested_chamber].set_xlabel(f"Propagated hit {direction} (mm)")
                rotation_axs[idirection][tested_chamber].set_ylabel(f"Residual {directions[idirection-1]} (mm)")
                rotation_axs[idirection][tested_chamber].set_title(f"BARI-0{tested_chamber+1}")

                # plot 2D distribution of residuals vs propagated position:
                residuals2d_xy_axs[idirection][tested_chamber].hist2d(prophits[idirection], residuals[idirection-1], bins=100, range=[[-40, 40],[-0.6, 0.6]])
                residuals2d_xy_axs[idirection][tested_chamber].set_title(f"BARI-0{tested_chamber+1}")
                residuals2d_xy_axs[idirection][tested_chamber].set_xlabel(f"Propagated position {direction} (mm)")
                residuals2d_xy_axs[idirection][tested_chamber].set_ylabel(f"Residual {directions[idirection-1]} (mm)")

                # plot 2D distribution of residuals vs propagated position:
                residuals2d_xx_axs[idirection][tested_chamber].hist2d(prophits[idirection], residuals[idirection], bins=100, range=[[-40, 40],[-0.6, 0.6]])
                residuals2d_xx_axs[idirection][tested_chamber].set_title(f"BARI-0{tested_chamber+1}")
                residuals2d_xx_axs[idirection][tested_chamber].set_xlabel(f"Propagated position {direction} (mm)")
                residuals2d_xx_axs[idirection][tested_chamber].set_ylabel(f"Residual {direction} (mm)")

                spres_axs[tested_chamber].plot(cluster_size_cuts, space_resolutions[direction], marker="o", label=direction)
                spres_axs[tested_chamber].text(
                    0.05, 0.9,
                    chamber_text.format(tested_chamber+1),
                    transform=spres_axs[tested_chamber].transAxes,
                    va="top", linespacing=1.7
                )
                spres_axs[tested_chamber].fill_between(
                    cluster_size_cuts,
                    np.array(space_resolutions[direction]) - np.array(err_space_resolutions[direction]),
                    np.array(space_resolutions[direction]) + np.array(err_space_resolutions[direction]),
                    alpha=0.2
                )
                hep.cms.text(text="Preliminary", ax=spres_axs[tested_chamber])
                spres_axs[tested_chamber].text(
                    1., 1., "CERN H4 test beam",
                    va="bottom", ha="right", weight="bold",
                    transform = spres_axs[tested_chamber].transAxes
                )
            #spres_axs[tested_chamber].plot(cluster_size_cuts, err_space_resolutions[direction], marker="o", label=direction)

            # bins_x = (matched_bins_x + 0.5*(matched_bins_x[1]-matched_bins_x[0]))[:-1]
            # bins_y = (matched_bins_y + 0.5*(matched_bins_y[1]-matched_bins_y[0]))[:-1]
            # #plt.contourf(bins_x, bins_y, efficiency)
            # plt.imshow(efficiency, extent=eff_range[0]+eff_range[1], origin="lower")
            # plt.xlabel("x (mm)")
            # plt.ylabel("y (mm)")
            # plt.colorbar(label="Efficiency")
            # plt.tight_layout()
            # plt.text(eff_range[0][-1]-.5, eff_range[1][-1]+2, "GEM-10x10-380XY-BARI-04", horizontalalignment="right")


            # plot propagation errors
            properr_axs[tested_chamber].hist(prophits_x_error[:,tested_chamber], bins=50, label="x", alpha=0.3, range=(0,1))
            properr_axs[tested_chamber].hist(prophits_y_error[:,tested_chamber], bins=50, label="y", alpha=0.3, range=(0,1))
            properr_axs[tested_chamber].set_xlabel("Extrapolation uncertainty (mm)")
            properr_axs[tested_chamber].set_title(f"BARI-0{tested_chamber+1}")
            properr_axs[tested_chamber].legend()

            spres_axs[tested_chamber].set_xlabel("Cluster size")            
            spres_axs[tested_chamber].set_ylabel(f"Residual sigma (µm)")
            #spres_axs[tested_chamber].set_title(f"BARI-0{tested_chamber+1}")
            spres_axs[tested_chamber].legend()

            h, binsx, binsy, img = cluster_size_axs[tested_chamber].hist2d(
                cluster_sizes[0], cluster_sizes[1],
                bins=10,
                range=((0.5, 10.5), (0.5, 10.5))
            )
            binsx_center = 0.5*(binsx[1:]+binsx[:-1])
            binsy_center = 0.5*(binsy[1:]+binsy[:-1])
            for ix, binx in enumerate(binsx_center):
                for iy, biny in enumerate(binsy_center):
                    cluster_size_axs[tested_chamber].text(
                        binx, biny, int(h[ix][iy]),
                        ha="center", va="center", size=11, weight="bold"
                    )
            cluster_size_fig.colorbar(img, ax=cluster_size_axs[tested_chamber], label="Events")
            cluster_size_axs[tested_chamber].set_xlabel("Cluster size x")
            cluster_size_axs[tested_chamber].set_ylabel("Cluster size y")
            cluster_size_axs[tested_chamber].set_title(f"BARI-0{tested_chamber+1}")

        print("Saving plots...")
        
        spres_fig.tight_layout()
        spres_fig.savefig(args.odir/"space_resolution.png")

        residual_fig.tight_layout()
        residual_fig.savefig(args.odir/"residuals.png")

        profile_fig.tight_layout()
        profile_fig.savefig(os.path.join(args.odir, "profile.png"))

        profile_fig.tight_layout()
        profile_fig.savefig(os.path.join(args.odir, "profile.png"))

        residual_cls_fig.tight_layout()
        residual_cls_fig.savefig(args.odir/"residuals_cls.png")

        properr_fig.tight_layout()
        properr_fig.savefig(args.odir/"extrapolation_error.png")
        
        rotation_fig.tight_layout()
        rotation_fig.savefig(args.odir/"rotation.png")

        prophits_fig.tight_layout()
        prophits_fig.savefig(args.odir/"prophits.png")

        residuals2d_xx_fig.tight_layout()
        residuals2d_xx_fig.savefig(args.odir/"residuals2d_xx.png")

        residuals2d_xy_fig.tight_layout()
        residuals2d_xy_fig.savefig(args.odir/"residuals2d_xy.png")

        cluster_size_fig.tight_layout()
        cluster_size_fig.savefig(args.odir/"cluster_size.png")

        chi2_fig.tight_layout()
        chi2_fig.savefig(args.odir/"chi2.png")

        # combine x and y angle corrections, then save:
        corrections = {
            "translation_x": translation[0], "translation_y": translation[1],
            "err_translation_x": err_translation[0], "err_translation_y": err_translation[1],
            "angle": np.sum(angles.T/err_angles.T**2, axis=1)/np.sum(1/err_angles.T**2, axis=1),
            "err_angle": np.sqrt(1/np.sum(1/err_angles.T**2, axis=1))
        }
        corrections_df = pd.DataFrame.from_dict(corrections)
        print("Corrections:\n", corrections_df)
        pd.DataFrame.from_dict(corrections).to_csv(args.odir / "corrections.txt", sep=";")

if __name__=='__main__': main()
