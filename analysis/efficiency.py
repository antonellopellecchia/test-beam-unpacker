""" Calculate and plot efficiency maps (2D) and profiles (1D) """

import os, sys, pathlib
import argparse
from tqdm import tqdm

import uproot
import numpy as np
import awkward as ak
import scipy
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mplhep as hep
plt.style.use(hep.style.ROOT)
plt.rcParams.update({'font.size': 32})

# enable multi-threading:
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()

def n_gauss(x, n, *args):
    m, s, k = args[0:n], args[n:n*2], args[n*2:n*3]
    return sum([
        k[i]*scipy.stats.norm.pdf(x, loc=m[i], scale=s[i])
        for i in range(n)
    ])

def gauss(x, *args):
    m, s, k = args
    return k*scipy.stats.norm.pdf(x, loc=m, scale=s)

def four_gauss(x, *args):
    return n_gauss(x, 4, *args)

def ten_gauss(x, *args):
    return n_gauss(x, 10, *args)

def nine_gauss(x, *args):
    return n_gauss(x, 9, *args)

def eight_gauss(x, *args):
    return n_gauss(x, 8, *args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ifile", type=pathlib.Path, help="Input file")
    parser.add_argument("odir", type=pathlib.Path, help="Output directory")
    parser.add_argument("detector", type=str, help="Detector under test")
    parser.add_argument("bins", type=int, help="Number of bins")
    parser.add_argument("-n", "--events", type=int, default=-1, help="Number of events to analyse")
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate logging")
    args = parser.parse_args()
    
    os.makedirs(args.odir, exist_ok=True)

    with uproot.open(args.ifile) as track_file:
        track_tree = track_file["trackTree"]
        if args.verbose: track_tree.show()

        print("Reading tree...")
        track_x_chi2 = track_tree["trackChi2X"].array(entry_stop=args.events)
        track_y_chi2 = track_tree["trackChi2Y"].array(entry_stop=args.events)
        chi2_fig, chi2_axs = plt.subplots(nrows=2, ncols=1, figsize=(10,9*2))
        chi2_axs[0].hist(track_x_chi2, range=(0.00001,30), bins=500, alpha=0.4)
        chi2_axs[0].hist(track_x_chi2[track_x_chi2<2], range=(0.00001,30), bins=500, alpha=0.4)
        chi2_axs[0].set_yscale("log")
        chi2_axs[0].set_xlabel("χ$^2_x$")
        chi2_axs[1].hist(track_y_chi2, range=(0.00001,30), bins=500, alpha=0.4)
        chi2_axs[1].hist(track_y_chi2[track_y_chi2<2], range=(0.00001,30), bins=500, alpha=0.4)
        chi2_axs[1].set_yscale("log")
        chi2_axs[1].set_xlabel("χ$^2_y$")
        chi2_fig.savefig(args.odir/"chi2.png")

        if args.detector=="ge21":
            rechit_chamber = track_tree["rechitChamber"].array(entry_stop=args.events)
            prophit_chamber = track_tree["prophitChamber"].array(entry_stop=args.events)
            #rechits_eta = track_tree["rechitEta"].array(entry_stop=args.events)
            #prophits_eta = ak.flatten(track_tree["prophitEta"].array(entry_stop=args.events))
            rechits_x = track_tree["rechitLocalX"].array(entry_stop=args.events)
            rechits_y = track_tree["rechitLocalY"].array(entry_stop=args.events)
            prophits_x = track_tree["prophitLocalX"].array(entry_stop=args.events)
            prophits_y = track_tree["prophitLocalY"].array(entry_stop=args.events)

            mask_chi2 = (track_x_chi2>0.1)&(track_x_chi2<2)&(track_y_chi2>0.1)&(track_y_chi2<2)
            rechit_chamber = rechit_chamber[mask_chi2]
            prophit_chamber = prophit_chamber[mask_chi2]
            rechits_x = rechits_x[mask_chi2]
            rechits_y = rechits_y[mask_chi2]
            prophits_x = prophits_x[mask_chi2]
            prophits_y = prophits_y[mask_chi2]
            track_x_chi2 = track_x_chi2[mask_chi2]
            track_y_chi2 = track_y_chi2[mask_chi2]

            ge21_chamber = 4
            prophits_x, prophits_y = ak.flatten(prophits_x[prophit_chamber==ge21_chamber]), ak.flatten(prophits_y[prophit_chamber==ge21_chamber])
            rechits_x, rechits_y = rechits_x[rechit_chamber==ge21_chamber], rechits_y[rechit_chamber==ge21_chamber]

            residuals_x, residuals_y = prophits_x-rechits_x, prophits_y-rechits_y

            print("Matching...")
            mask_out = (abs(prophits_x)<40.)&(abs(prophits_y)<40.)
            rechits_x, rechits_y = rechits_x[mask_out], rechits_y[mask_out]
            prophits_x, prophits_y = prophits_x[mask_out], prophits_y[mask_out]

            matches = ak.count(rechits_x, axis=1)>0
            matched_x, matched_y = prophits_x[matches], prophits_y[matches]


            """ Plot 2D efficiency map: """
            print("Calculating efficiency map...")
            eff_fig, eff_ax = plt.figure(figsize=(12,9)), plt.axes()
            eff_range = [[min(prophits_x), max(prophits_x)], [min(prophits_y), max(prophits_y)]]
            matched_histogram, matched_bins_x, matched_bins_y = np.histogram2d(matched_x, matched_y, bins=args.bins, range=eff_range)
            total_histogram, total_bins_x, total_bins_y = np.histogram2d(prophits_x, prophits_y, bins=args.bins, range=eff_range)

            if not (np.array_equal(matched_bins_x,total_bins_x) and np.array_equal(matched_bins_y,total_bins_y)):
                raise ValueError("Different bins between numerator and denominator")
            efficiency = np.divide(matched_histogram, total_histogram, where=(total_histogram!=0))

            centers_x = 0.5*(matched_bins_x[1:]+matched_bins_x[:-1])
            centers_y = 0.5*(matched_bins_y[1:]+matched_bins_y[:-1])
            print(efficiency)
            average_efficiency = efficiency.flatten().mean()
            print("Average efficiency", efficiency.flatten().mean())
            
            print("Plotting efficiency map...")
            img = eff_ax.imshow(
                efficiency,
                extent=[matched_bins_x[0], matched_bins_x[-1], matched_bins_y[0], matched_bins_y[-1]],
                origin="lower"
            )
            eff_ax.set_xlabel("Extrapolated x (mm)")
            eff_ax.set_ylabel("Extrapolated y (mm)")
            eff_ax.set_title(
                r"$\bf{CMS}\,\,\it{Preliminary}$",
                color="black", weight="normal", loc="left", size=28
            )
            eff_fig.colorbar(img, ax=eff_ax, label="Efficiency")
            img.set_clim(.85, 1.)
            eff_fig.tight_layout()
            eff_ax.text(1.0, 1.01, "GE2/1 H4 test beam", transform=eff_ax.transAxes, ha="right", weight="bold", size=28)
            print("Saving result...")
            eff_fig.savefig(os.path.join(args.odir, "ge21.png"))
            eff_fig.savefig(os.path.join(args.odir, "ge21.pdf"))

            
            """ Plot efficiency profile in 1D vs y position """
            print("Calculating 1D efficiency profile...")
            eff_fig, eff_ax = plt.figure(figsize=(11,9)), plt.axes()
            # matched_histogram, matched_bins_y = np.histogram(matched_y, bins=500)
            # total_histogram, total_bins_y = np.histogram(prophits_y, bins=500)
            # efficiency_1d = np.divide(matched_histogram, total_histogram, where=(total_histogram!=0))
            efficiency_1d = efficiency.mean(axis=1)
            print("Plotting 1D efficiency profile...")
            centers_y = 0.5*(matched_bins_y[1:]+matched_bins_y[:-1])

            # plot and fit with four gaussians:
            print("Fitting efficiency")          
            eff_ax.plot(centers_y, efficiency_1d, ".k")
            params = [
                -20, 0, 20, 40, # means
                1, 1, 1, 1, # sigma
                0.7, 0.7, 0.7, 0.7 # constants 
            ]
            fit_function = lambda x, *args: 1 - four_gauss(x, *args)
            params, cov = scipy.optimize.curve_fit(
                fit_function, centers_y, efficiency_1d, p0=params
            )
            perr = np.sqrt(np.diag(cov))
            x = np.linspace(centers_y[0], centers_y[-1], 1000)
            efficiency_interp = fit_function(x, *params)
            eff_ax.plot(x, efficiency_interp, color="red")
            eff_ax.set_xlim(-50, 50)
            eff_ax.set_ylim(0.0, 1.1)
            m, s, k = params[0:4], params[4:8], params[8:12]
            err_m, err_s, err_k = perr[0:4], perr[4:8], perr[8:12]
            print("means", m, "\nsigma", s, "\nconstant", k)
            for i in range(4):
                eff_ax.text(
                    m[i], fit_function(m[i], *params)-0.1,
                    f"$\sigma$ = {s[i]*1e3:1.0f} $\pm$ {err_s[i]*1e3:1.0f} µm", size=15, rotation=30, ha="center"
                )
                #eff_ax.text(m[i]-7, 1-2.2*k[i], f"$\sigma$ = {s[i]*1e3:1.1f} µm", size=15)

            below_97 = efficiency_interp[efficiency_interp<0.97]
            below_97_x = x[efficiency_interp<0.97]
            print(below_97)
            below_97_index = np.argpartition(below_97, 8)[-8:]
            below_97_positions = np.sort(below_97_x[below_97_index])
            print(below_97_index)
            print(below_97_positions)
            print("Below 97:", below_97_positions)
            below_97_widths = below_97_positions[1::2]-below_97_positions[::2]
            print(below_97_widths)
            #eff_ax.plot(x, 0.97+np.zeros(x.size), "-.", color="blue")
            #eff_ax.plot(x, average_efficiency+np.zeros(x.size), "-.", color="blue")
            for i in range(4):
                eff_ax.text(
                    m[i]-1, fit_function(m[i], *params)-0.25,
                    #f"{below_97_widths[i]*1e3:1.1f} µm at 97%",
                    " ",
                    size=15, rotation=60
                )

            eff_ax.set_xlabel("Extrapolated y (mm)")
            eff_ax.set_ylabel("Efficiency")
            eff_ax.set_title(
                r"$\bf{CMS}\,\,\it{Preliminary}$",
                color='black', weight='normal', loc="left", size=28
            )
            eff_ax.text(1., 1., "GE2/1 H4 test beam", transform=eff_ax.transAxes, ha="right", va="bottom", weight="bold", size=28)
            eff_fig.tight_layout()
            print("Saving result...")
            eff_fig.savefig(os.path.join(args.odir, "ge21_1d.png"))


            """ Calculate angular alignment based on HV sectors """
            # choose only points close to 1 sector:
            centers_x = 0.5*(matched_bins_x[1:]+matched_bins_x[:-1])
            centers_y = 0.5*(matched_bins_y[1:]+matched_bins_y[:-1])
            map_mask = centers_y < -10
            centers_y = centers_y[map_mask]
            efficiency = efficiency[map_mask]

            # slice efficiency map along 10 y points:
            npoints_x = 10
            step_x = int(ak.count(centers_x)/npoints_x)
            slices_x = centers_x.T[::step_x].T
            efficiency_slices = efficiency.T[::step_x]

            # calculate min position for each slice:
            slices_fig, slices_ax = plt.figure(), plt.subplot(projection="3d")
            min_positions = list()
            for slice_x, eff_slice in zip(slices_x, efficiency_slices):
                slices_ax.plot(centers_y, slice_x*np.ones(ak.count(centers_y)), eff_slice)
                eff_min = ak.min(eff_slice)
                y_min = centers_y[eff_slice==eff_min] # where efficiency minimum is
                min_positions.append(y_min[0])
            slices_ax.set_xlabel("Extrapolated y (mm)")
            slices_ax.set_ylabel("Extrapolated x (mm)")
            slices_ax.set_zlabel("Efficiency")
            slices_fig.savefig(os.path.join(args.odir, "ge21_slices.png"))

            # fit linearly:
            def linear_func(x, m, q): return m*x+q
            (m, q), cov =  curve_fit(linear_func, np.array(slices_x), min_positions)
            err_m, err_q = np.sqrt(np.diag(cov))
            theta = np.arcsin(m)
            err_theta = err_m/np.sqrt(1-m**2)
            print("Theta:", theta, "+-", err_theta, "rad")

            # plot correlation:
            rotation_fig, rotation_ax = plt.figure(), plt.axes()
            rotation_ax.plot(slices_x, min_positions, "o") # plot data
            rotation_ax.plot(slices_x, linear_func(slices_x, m, q), color="red") # plot fit
            rotation_ax.text(
                0.42, 0.87,
                f"$\\theta$ = {theta*1e3:1.1f} $\pm$ {err_theta*1e3:1.1f} mrad",
                transform=rotation_ax.transAxes,
                bbox=dict(boxstyle="square, pad=0.5", ec="black", fc="none")
            )
            rotation_ax.text(.87, 1.01, "GE2/1", transform=rotation_ax.transAxes)
            rotation_ax.set_xlabel("Extrapolated x (mm)")
            rotation_ax.set_ylabel("Displacement (mm)")
            rotation_fig.savefig(os.path.join(args.odir, "ge21_rotation.png"))
            

        if args.detector=="me0":
            rechit_chamber = track_tree["rechitChamber"].array(entry_stop=args.events)
            prophit_chamber = track_tree["prophitChamber"].array(entry_stop=args.events)
            #rechits_eta = track_tree["rechitEta"].array(entry_stop=args.events)
            #prophits_eta = ak.flatten(track_tree["prophitEta"].array(entry_stop=args.events))
            rechits_x = track_tree["rechitLocalX"].array(entry_stop=args.events)
            rechits_y = track_tree["rechitLocalY"].array(entry_stop=args.events)
            prophits_x = track_tree["prophitLocalX"].array(entry_stop=args.events)
            prophits_y = track_tree["prophitLocalY"].array(entry_stop=args.events)

            mask_chi2 = (track_x_chi2>0.1)&(track_x_chi2<2)&(track_y_chi2>0.1)&(track_y_chi2<2)
            rechit_chamber = rechit_chamber[mask_chi2]
            prophit_chamber = prophit_chamber[mask_chi2]
            rechits_x = rechits_x[mask_chi2]
            rechits_y = rechits_y[mask_chi2]
            prophits_x = prophits_x[mask_chi2]
            prophits_y = prophits_y[mask_chi2]
            track_x_chi2 = track_x_chi2[mask_chi2]
            track_y_chi2 = track_y_chi2[mask_chi2]

            me0_chamber = 5
            prophits_x, prophits_y = ak.flatten(prophits_x[prophit_chamber==me0_chamber]), ak.flatten(prophits_y[prophit_chamber==me0_chamber])
            rechits_x, rechits_y = rechits_x[rechit_chamber==me0_chamber], rechits_y[rechit_chamber==me0_chamber]

            print("Plotting chi2 distributions...")
            chi2_position_fig, chi2_position_axs = plt.subplots(nrows=2, ncols=2, figsize=(10*2,9*2))
            for i,(coord_prop,prop) in enumerate(zip(["x","y"], [prophits_x, prophits_y])):
                for j,(coord_chi2,chi2) in enumerate(zip(["x","y"], [track_x_chi2, track_y_chi2])):
                    chi2_position_axs[i][j].hist2d(chi2, prop, range=((0.1,20), (-40,40)), bins=100)
                    chi2_position_axs[i][j].set_xlabel("χ$^2_" + coord_chi2 + "$")
                    chi2_position_axs[i][j].set_ylabel("propagated " + coord_prop + " (mm)")
            chi2_position_fig.tight_layout()
            chi2_position_fig.savefig(args.odir/"me0_chi2.png")

            print("Matching...")
            mask_out = (abs(prophits_x)<40.)&(abs(prophits_y)<40.)
            rechits_x, rechits_y = rechits_x[mask_out], rechits_y[mask_out]
            prophits_x, prophits_y = prophits_x[mask_out], prophits_y[mask_out]
            matches = ak.count(rechits_x, axis=1)>0 # only check whether there is a rechit in the event
            matched_x, matched_y = prophits_x[matches], prophits_y[matches]

            print("Calculating efficiency map...")
            eff_fig, eff_ax = plt.figure(figsize=(12,9)), plt.axes()
            eff_range = [[min(prophits_x), max(prophits_x)], [min(prophits_y), max(prophits_y)]]
            matched_histogram, matched_bins_x, matched_bins_y = np.histogram2d(matched_x, matched_y, bins=args.bins, range=eff_range)
            total_histogram, total_bins_x, total_bins_y = np.histogram2d(prophits_x, prophits_y, bins=args.bins, range=eff_range)

            print(matched_histogram)
            print(total_histogram)
            print(ak.count(matched_bins_x), matched_bins_x)
            print(ak.count(matched_bins_y), matched_bins_y)

            if not (np.array_equal(matched_bins_x,total_bins_x) and np.array_equal(matched_bins_y,total_bins_y)):
                raise ValueError("Different bins between numerator and denominator")
            efficiency = np.divide(matched_histogram, total_histogram, where=(total_histogram!=0))

            print(ak.count(efficiency), efficiency)
            average_efficiency = efficiency.flatten().mean()
            print(efficiency.flatten().mean())
            
            print("Plotting efficiency map...")
            img = eff_ax.imshow(
                efficiency,
                extent=[matched_bins_x[0], matched_bins_x[-1], matched_bins_y[0], matched_bins_y[-1]],
                origin="lower"
            )
            eff_ax.set_xlabel("Extrapolated x (mm)")
            eff_ax.set_ylabel("Extrapolated y (mm)")
            eff_ax.set_title(
                r"$\bf{CMS}\,\,\it{Preliminary}$",
                color='black', weight='normal', loc="left", size=28
            )
            # cax = eff_fig.add_axes([
            #     eff_ax.get_position().x1+0.01,
            #     eff_ax.get_position().y0,
            #     0.02,eff_ax.get_position().height
            # ])
            eff_fig.colorbar(img, ax=eff_ax, label="Efficiency")
            img.set_clim(.85, 1.)
            eff_fig.tight_layout()
            eff_ax.text(1., 1., "ME0 H4 test beam", transform=eff_ax.transAxes, ha="right", va="bottom", weight="bold", size=28)
            print("Saving result...")
            eff_fig.savefig(os.path.join(args.odir, "me0.png"))
            eff_fig.savefig(os.path.join(args.odir, "me0.pdf"))

            
            """ Plot 1D efficiency profile vs y """
            # choose only points close to 1 sector:
            centers_x = 0.5*(matched_bins_x[1:]+matched_bins_x[:-1])
            centers_y = 0.5*(matched_bins_y[1:]+matched_bins_y[:-1])
            #centers_x, centers_y = centers_x[mask_x], centers_y[mask_y]
            #efficiency = efficiency[map_mask]

            # choose only points in (-35,35)x(-30,20)
            mask_x = (centers_x>-35)&(centers_x<28)
            mask_y = (centers_x>-30)&(centers_x<20)
            # slice efficiency map along 10 y points:
            npoints_y = 10
            step_y = int(np.ceil(centers_y[mask_y].size/npoints_y))
            slices_y = centers_y[mask_y][::step_y]
            efficiency_slices = efficiency[mask_y][::step_y]
            #efficiency_slices = efficiency_slices[mask_x]

            # plot each slice:
            #slices_fig, slices_axs = plt.subplots(nrows=npoints_y, ncols=1, figsize=(9, 8*(npoints_y)))
            for i_slice,(slice_y,eff_slice) in enumerate(zip(slices_y, efficiency_slices)):
                # plot and fit with ten gaussians:
                slices_fig, slices_axs = plt.figure(figsize=(12,10)), plt.axes()
                eff_slice = eff_slice[mask_x]
                # print(centers_x[mask_x], mask_x, eff_slice)
                # print(ak.count(centers_x, axis=0), ak.count(mask_x, axis=0), ak.count(eff_slice, axis=0))
                slices_axs.plot(centers_x[mask_x], eff_slice, ".k")

                print(f"Fitting efficiency for y={slice_y:1.2f} mm")          
                params = [
                    -30, -20, -15, -10, 0, 5, 15, 20, # means
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, # sigma
                    0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7 # constants 
                ]
                fit_function = lambda x, *args: 1 - eight_gauss(x, *args)
                try:
                    params, cov = scipy.optimize.curve_fit(
                        fit_function, centers_x[mask_x], eff_slice, p0=params
                    )
                    perr = np.sqrt(np.diag(cov))

                except RuntimeError: print("Skipping, fit failed...")
                x = np.linspace(centers_x[mask_x][0], centers_x[mask_x][-1], 1000000)
                efficiency_interp = fit_function(x, *params)
                slices_axs.plot(x, efficiency_interp, color="red")

                m, s, k = params[0:8], params[8:16], params[16:24]
                err_m, err_s, err_k = perr[0:8], perr[8:16], perr[16:24]
                print("means", m, "\nsigma", s, "\nconstant", k)
                #slices_axs.plot(x, average_efficiency+np.zeros(x.size), ".-", color="blue")
                slice_sigmas, slice_sigma_errs = list(), list()
                for i in range(8):
                    #slices_axs.text(
                    #    m[i], fit_function(m[i], *params)-0.1,
                    #    f"$\sigma$ = {s[i]*1e3:1.0f} $\pm$ {err_s[i]*1e3:1.0f} µm", size=14, rotation=30, ha="center"
                    #)
                    width_97 = 4*s[i]
                    #width_97 = 2 * s[i] * np.sqrt( -2 * np.log(0.3 * s[i] * np.sqrt(2*np.pi)/k[i]) )
                    # slices_axs[i_slice].plot(m[i]-0.5*width_97, fit_function(m[i]-0.5*width_97, *params), "o", markersize=5, color="blue")
                    # slices_axs[i_slice].plot(m[i]+0.5*width_97, fit_function(m[i]+0.5*width_97, *params), "o", markersize=5, color="blue")
                    slices_axs.text(
                        m[i]-1, fit_function(m[i], *params)-0.25,
                        " ",
                        #f"{width_97*1e3:1.1f} µm at 97%",
                        size=14, rotation=60, color="blue"
                    )

                s_mean = sum(s)/len(s)
                err_s_mean = sum(err_s)/len(err_s)
                m_mean, k_mean = sum(m)/len(m), sum(k)/len(k)
                print(f"Mean m {m_mean}, mean k {1-1.2*k_mean}, mean s {s_mean}")
                slices_axs.text(m_mean, 1-2.7*k_mean, f"average dip $\sigma$ = {s_mean*1e3:1.0f} $\pm$ {err_s_mean*1e3:1.0f} µm", size=32, ha="center")
                
                # below_97 = efficiency_interp[efficiency_interp<0.97]
                # below_97_x = x[efficiency_interp<0.97]
                # print(below_97)
                # below_97_index = np.argpartition(below_97, 16)[-16:]
                # below_97_positions = np.sort(below_97_x[below_95_index])
                # print(below_97_index)
                # print("Below 97:", below_97_positions)
                # print("Efficiency:", fit_function(below_97_positions, *params))
                # below_97_widths = below_97_positions[1::2]-below_95_positions[::2]
                # print(below_97_widths)

                slices_axs.set_xlim(-40, 30)
                slices_axs.set_ylim(0.0, 1.1)
                slices_axs.set_xlabel("Extrapolated x (mm)")
                slices_axs.set_ylabel("Efficiency")
                slices_axs.text(1.0, 1.0, "ME0 H4 test beam", transform=slices_axs.transAxes, ha="right", va="bottom", weight="bold")
                hep.cms.text(text="Preliminary", ax=slices_axs)
                #slices_axs[i_slice].set_title(f"y = {slice_y:1.2f} mm")
                slices_fig.savefig(os.path.join(args.odir, f"me0_slices_{i_slice}.png"))
                slices_fig.savefig(os.path.join(args.odir, f"me0_slices_{i_slice}.pdf"))
            slices_fig.tight_layout()
            slices_fig.savefig(os.path.join(args.odir, "me0_slices.png"))
            slices_fig.savefig(os.path.join(args.odir, "me0_slices.pdf"))


        if args.detector=="20x10":
            rechit_chamber = track_tree["rechitChamber"].array(entry_stop=args.events)
            prophit_chamber = track_tree["prophitChamber"].array(entry_stop=args.events)
            # rechits_eta = track_tree["rechitEta"].array(entry_stop=args.events)
            # prophits_eta = ak.flatten(track_tree["prophitEta"].array(entry_stop=args.events))
            rechits_x = track_tree["rechitLocalX"].array(entry_stop=args.events)
            rechits_y = track_tree["rechitLocalY"].array(entry_stop=args.events)
            prophits_x = track_tree["prophitLocalX"].array(entry_stop=args.events)
            prophits_y = track_tree["prophitLocalY"].array(entry_stop=args.events)

            mask_chi2 = (track_x_chi2>0.1)&(track_x_chi2<2)&(track_y_chi2>0.1)&(track_y_chi2<2)
            rechit_chamber = rechit_chamber[mask_chi2]
            prophit_chamber = prophit_chamber[mask_chi2]
            rechits_x = rechits_x[mask_chi2]
            rechits_y = rechits_y[mask_chi2]
            prophits_x = prophits_x[mask_chi2]
            prophits_y = prophits_y[mask_chi2]
            track_x_chi2 = track_x_chi2[mask_chi2]
            track_y_chi2 = track_y_chi2[mask_chi2]

            rectangular_chamber = 6
            prophits_x, prophits_y = ak.flatten(prophits_x[prophit_chamber==rectangular_chamber]), ak.flatten(prophits_y[prophit_chamber==rectangular_chamber])
            rechits_x, rechits_y = rechits_x[rechit_chamber==rectangular_chamber], rechits_y[rechit_chamber==rectangular_chamber]

            print("Plotting chi2 distributions...")
            chi2_position_fig, chi2_position_axs = plt.subplots(nrows=2, ncols=2, figsize=(10*2,9*2))
            for i,(coord_prop,prop) in enumerate(zip(["x","y"], [prophits_x, prophits_y])):
                for j,(coord_chi2,chi2) in enumerate(zip(["x","y"], [track_x_chi2, track_y_chi2])):
                    chi2_position_axs[i][j].hist2d(chi2, prop, range=((0.1,20), (-40,40)), bins=100)
                    chi2_position_axs[i][j].set_xlabel("χ$^2_" + coord_chi2 + "$")
                    chi2_position_axs[i][j].set_ylabel("propagated " + coord_prop + " (mm)")
            chi2_position_fig.tight_layout()
            chi2_position_fig.savefig(args.odir/"20x10_chi2.png")

            print("Matching...")
            mask_out = (abs(prophits_x)<40.)&(abs(prophits_y)<40.)
            rechits_x, rechits_y = rechits_x[mask_out], rechits_y[mask_out]
            prophits_x, prophits_y = prophits_x[mask_out], prophits_y[mask_out]
            matches = ak.count(rechits_x, axis=1)>0
            matched_x, matched_y = prophits_x[matches], prophits_y[matches]

            print("Calculating efficiency map...")
            eff_fig, eff_ax = plt.figure(figsize=(12,9)), plt.axes()
            eff_range = [[min(prophits_x), max(prophits_x)], [min(prophits_y), max(prophits_y)]]
            matched_histogram, matched_bins_x, matched_bins_y = np.histogram2d(matched_x, matched_y, bins=args.bins, range=eff_range)
            total_histogram, total_bins_x, total_bins_y = np.histogram2d(prophits_x, prophits_y, bins=args.bins, range=eff_range)

            print(matched_histogram)
            print(total_histogram)
            print(ak.count(matched_bins_x), matched_bins_x)
            print(ak.count(matched_bins_y), matched_bins_y)

            if not (np.array_equal(matched_bins_x,total_bins_x) and np.array_equal(matched_bins_y,total_bins_y)):
                raise ValueError("Different bins between numerator and denominator")
            efficiency = np.divide(matched_histogram, total_histogram, where=(total_histogram!=0))

            print(ak.count(efficiency), efficiency)
            print("Plotting efficiency map...")
            img = eff_ax.imshow(
                efficiency,
                extent=[matched_bins_x[0], matched_bins_x[-1], matched_bins_y[0], matched_bins_y[-1]],
                origin="lower"
            )
            eff_ax.set_xlabel("Extrapolated x (mm)")
            eff_ax.set_ylabel("Extrapolated y (mm)")
            eff_ax.set_title(
                r"$\bf{CMS}\,\,\it{Preliminary}$",
                color='black', weight='normal', loc="left", size=28
            )
            eff_fig.colorbar(img, ax=eff_ax, label="Efficiency")
            img.set_clim(.85, 1.)
            eff_fig.tight_layout()
            eff_ax.text(1., 1., "$20\\times 10\,cm^2$", transform=eff_ax.transAxes, ha="right", va="bottom", weight="bold", size=28)
            print("Saving result...")
            eff_fig.savefig(os.path.join(args.odir, "20x10.png"))
            eff_fig.savefig(os.path.join(args.odir, "20x10.pdf"))

            
            """ Calculate angular alignment based on HV sectors """
            # choose only points close to 1 sector:
            centers_x = 0.5*(matched_bins_x[1:]+matched_bins_x[:-1])
            centers_y = 0.5*(matched_bins_y[1:]+matched_bins_y[:-1])
            map_mask = centers_y > 10
            centers_y = centers_y[map_mask]
            efficiency = efficiency[map_mask]

            # slice efficiency map along 10 y points:
            npoints_y = 10
            step_y = int(ak.count(centers_y)/npoints_y)
            slices_y = centers_y[::step_y]
            efficiency_slices = efficiency[::step_y]

            mask_x = abs(centers_x)<5
            centers_x = centers_x[mask_x]
            # calculate min position for each slice:
            slices_fig, slices_ax = plt.figure(), plt.subplot(projection="3d")
            min_positions = list()
            for slice_y, eff_slice in zip(slices_y, efficiency_slices):
                eff_slice = eff_slice[mask_x]
                slices_ax.plot(centers_x, slice_y*np.ones(ak.count(centers_x)), eff_slice)
                eff_min = ak.min(eff_slice)
                x_min = centers_x[eff_slice==eff_min] # where efficiency minimum is
                min_positions.append(x_min[0])
                print(x_min, eff_min)
            slices_ax.set_xlabel("Extrapolated x (mm)")
            slices_ax.set_ylabel("Extrapolated y (mm)")
            slices_ax.set_zlabel("Efficiency")
            slices_fig.savefig(os.path.join(args.odir, "20x10_slices.png"))

            # fit linearly:
            def linear_func(x, m, q): return m*x+q
            (m, q), cov =  curve_fit(linear_func, np.array(slices_y), min_positions)
            err_m, err_q = np.sqrt(np.diag(cov))
            theta = np.arcsin(m)
            err_theta = err_m/np.sqrt(1-m**2)
            print("Theta:", theta, "+-", err_theta, "rad")

            # plot correlation:
            rotation_fig, rotation_ax = plt.figure(), plt.axes()
            rotation_ax.plot(slices_y, min_positions, "o") # plot data
            rotation_ax.plot(slices_y, linear_func(slices_y, m, q), color="red") # plot fit
            rotation_ax.text(
                0.42, 0.87,
                f"$\\theta$ = {theta*1e3:1.1f} $\pm$ {err_theta*1e3:1.1f} mrad",
                transform=rotation_ax.transAxes,
                bbox=dict(boxstyle="square, pad=0.5", ec="black", fc="none")
            )
            rotation_ax.text(.87, 1.01, "20x10", transform=rotation_ax.transAxes)
            rotation_ax.set_xlabel("Extrapolated y (mm)")
            rotation_ax.set_ylabel("Displacement (mm)")
            rotation_fig.savefig(os.path.join(args.odir, "20x10_rotation.png"))


            """ Plot efficiency profile in 1D vs y position """
            print("Calculating 1D efficiency profile...")
            eff_fig, eff_ax = plt.figure(figsize=(11,9)), plt.axes()
            # matched_histogram, matched_bins_y = np.histogram(matched_y, bins=500)
            # total_histogram, total_bins_y = np.histogram(prophits_y, bins=500)
            # efficiency_1d = np.divide(matched_histogram, total_histogram, where=(total_histogram!=0))
            efficiency_1d = efficiency.mean(axis=0)
            print("Plotting 1D efficiency profile...")
            centers_x = 0.5*(matched_bins_x[1:]+matched_bins_x[:-1])

            # plot and fit with four gaussians:
            print("Fitting efficiency")          
            eff_ax.plot(centers_x, efficiency_1d, ".k")
            params = [ 0, 1, 0.2 ]
            fit_function = lambda x, *args: 1 - gauss(x, *args)
            params, cov = scipy.optimize.curve_fit(
                fit_function, centers_x, efficiency_1d, p0=params
            )
            perr = np.sqrt(np.diag(cov))
            x = np.linspace(centers_x[0], centers_x[-1], 1000)
            eff_ax.plot(x, fit_function(x, *params), color="red")
            eff_ax.set_ylim(0.9, 1.01)
            m, s, k = params
            err_m, err_s, err_k = perr
            print("mean", m, "\nsigma", s, "\nconstant", k)
            eff_ax.text(m, 1-1.2*k, f"$\sigma$ = {s*1e3:1.0f} $\pm$ {err_s*1e3:1.0f} µm", size=32, ha="center")

            eff_ax.set_xlabel("Extrapolated x (mm)")
            eff_ax.set_ylabel("Efficiency")
            eff_ax.set_title(
                r"$\bf{CMS}\,\,\it{Preliminary}$",
                color='black', weight='normal', loc="left", size=28
            )
            eff_ax.text(1., 1., "$20\\times 10\,cm^2$ H4 test beam", transform=eff_ax.transAxes, ha="right", va="bottom", weight="bold", size=28)
            eff_fig.tight_layout()
            print("Saving result...")
            eff_fig.savefig(os.path.join(args.odir, "20x10_1d.png"))
            eff_fig.savefig(os.path.join(args.odir, "20x10_1d.pdf"))

        elif args.detector=="tracker":
            rechits_chamber = track_tree["rechits2D_Chamber"].array(entry_stop=args.events)
            rechits_x = track_tree["rechits2D_X"].array(entry_stop=args.events)
            rechits_y = track_tree["rechits2D_Y"].array(entry_stop=args.events)
            prophits_x = track_tree["prophits2D_X"].array(entry_stop=args.events)
            prophits_y = track_tree["prophits2D_Y"].array(entry_stop=args.events)

            print("chambers", rechits_chamber)
            print("prophits", prophits_x)
            print("rechits", rechits_x)

            eff_fig, eff_axs = plt.subplots(nrows=1, ncols=4, figsize=(48,9))
            for tested_chamber in range(4):
                prophits_x_chamber = prophits_x[:,tested_chamber]
                prophits_y_chamber = prophits_y[:,tested_chamber]

                # stay in chamber boundary:
                mask_out = (abs(prophits_x_chamber)<40.)&(abs(prophits_y_chamber)<40.)
                rechits_chamber_inside = rechits_chamber[mask_out]
                prophits_x_chamber, prophits_y_chamber = prophits_x_chamber[mask_out], prophits_y_chamber[mask_out]
                rechits_x_chamber, rechits_y_chamber = rechits_x[mask_out], rechits_y[mask_out]

                # choose only events with at least 3 hits:
                mask_3hit = ak.count_nonzero(rechits_chamber_inside>=0, axis=1)>2
                rechits_chamber_inside = rechits_chamber_inside[mask_3hit]
                rechits_x_chamber, rechits_y_chamber = rechits_x_chamber[mask_3hit], rechits_y_chamber[mask_3hit]
                prophits_x_chamber, prophits_y_chamber = prophits_x_chamber[mask_3hit], prophits_y_chamber[mask_3hit]

                # list only events with a rechit in tested chamber, otherwise None:
                rechit_mask = ak.count_nonzero(rechits_chamber_inside==tested_chamber, axis=1)>0
                rechits_x_chamber, rechits_y_chamber = rechits_x_chamber.mask[rechit_mask], rechits_y_chamber.mask[rechit_mask]

                # for good events, list only rechit in tested chamber
                chamber_mask = rechits_chamber_inside==tested_chamber
                # to refine matching with angle later:
                # rechits_x_chamber = ak.sum(rechits_x_chamber.mask[chamber_mask], axis=1)
                # rechits_y_chamber = ak.sum(rechits_y_chamber.mask[chamber_mask], axis=1)
                chamber_mask = chamber_mask[chamber_mask]
                matched_x_chamber = prophits_x_chamber[chamber_mask]
                matched_y_chamber = prophits_y_chamber[chamber_mask]

                print("Calculating efficiency map...")
                eff_range = [[min(prophits_x_chamber), max(prophits_x_chamber)], [min(prophits_y_chamber), max(prophits_y_chamber)]]
                matched_histogram, matched_bins_x, matched_bins_y = np.histogram2d(matched_x_chamber, matched_y_chamber, bins=args.bins, range=eff_range)
                total_histogram, total_bins_x, total_bins_y = np.histogram2d(prophits_x_chamber, prophits_y_chamber, bins=args.bins, range=eff_range)

                if not (np.array_equal(matched_bins_x,total_bins_x) and np.array_equal(matched_bins_y,total_bins_y)):
                    raise ValueError("Different bins between numerator and denominator")
                efficiency = np.divide(matched_histogram, total_histogram, where=(total_histogram!=0))
                print(efficiency)
                
                print("Plotting efficiency map...")
                # bins_x = (matched_bins_x + 0.5*(matched_bins_x[1]-matched_bins_x[0]))[:-1]
                # bins_y = (matched_bins_y + 0.5*(matched_bins_y[1]-matched_bins_y[0]))[:-1]
                #plt.contourf(bins_x, bins_y, efficiency)
                img = eff_axs[tested_chamber].imshow(
                    efficiency,
                    #extent = [0, 1, 0, 1],
                    #extent=eff_range[0]+eff_range[1],
                    extent=[matched_bins_x[0], matched_bins_x[-1], matched_bins_y[0], matched_bins_y[-1], ],
                    origin="lower"
                )
                eff_axs[tested_chamber].set_xlabel("Extrapolated x (mm)")
                eff_axs[tested_chamber].set_ylabel("Extrapolated y (mm)")
                eff_axs[tested_chamber].set_title(
                    r"$\bf{CMS}\,\,\it{Preliminary}$",
                    color='black', weight='normal', loc="left"
                )
                eff_fig.colorbar(img, ax=eff_axs[tested_chamber], label="Efficiency")
                img.set_clim(.85, 1.)
                eff_axs[tested_chamber].text(eff_range[0][-1]-.5, eff_range[1][-1]+2, f"BARI-0{tested_chamber+1}", horizontalalignment="right")

            print("Saving result...")
            eff_fig.tight_layout()
            eff_fig.savefig(os.path.join(args.odir, "tracker.png"))

if __name__=="__main__": main()
