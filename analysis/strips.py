import os, sys, pathlib
import argparse
from tqdm import tqdm

import uproot
import numpy as np
import awkward as ak

import scipy
import scipy.stats
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

chamber_text = "GE2/1-II-M1-CERN-0001\n"\
    +"Effective gas gain $2\\times10^4$\n"\
    +"Ar-$CO_2$ 70%-30%\n"\
    +"Strip pitch 952 µrad\n"\
    +"150 GeV muons\n"\

def linear_function(x, q, m):
    return q + m*x

def gauss(x, *p):
    A, mu, sigma = p
    #return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    return A * scipy.stats.norm.pdf(x, loc=mu, scale=sigma)

def gauss2(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*scipy.stats.norm.pdf(x, loc=mu1, scale=sigma1) + \
    A2*scipy.stats.norm.pdf(x, loc=mu2, scale=sigma2)
    #return gauss(x, A1, mu1, sigma1) + gauss(x, A2, mu2, sigma2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ifile", type=pathlib.Path, help="Input file")
    parser.add_argument('odir', type=pathlib.Path, help="Output directory")
    parser.add_argument("--theta", type=float, default=0.0, help="Angular correction")
    parser.add_argument("-n", "--events", type=int, default=-1, help="Number of events to analyse")
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate logging")
    parser.add_argument("--chamber", type=int, default=4, help="Tested chamber (default GE2/1)")
    parser.add_argument("--alignment", type=float, nargs="+", default=[0, 0, 0], help="x, y, angle")
    args = parser.parse_args()
    
    os.makedirs(args.odir, exist_ok=True)

    with uproot.open(args.ifile) as track_file:
        track_tree = track_file["trackTree"]
        if args.verbose: track_tree.show()

        print("Reading tree...")
        # 1D branches:
        track_x_chi2 = track_tree["trackChi2X"].array(entry_stop=args.events)
        track_y_chi2 = track_tree["trackChi2Y"].array(entry_stop=args.events)
        rechit_chamber = track_tree["rechitChamber"].array(entry_stop=args.events)
        prophit_chamber = track_tree["prophitChamber"].array(entry_stop=args.events)
        rechits_eta = track_tree["rechitEta"].array(entry_stop=args.events)
        #prophits_eta = track_tree["prophitEta"].array(entry_stop=args.events)
        rechits_x = track_tree["rechitGlobalX"].array(entry_stop=args.events)
        rechits_y = track_tree["rechitGlobalY"].array(entry_stop=args.events)
        rechits_local_x = track_tree["rechitLocalX"].array(entry_stop=args.events)
        rechits_local_y = track_tree["rechitLocalY"].array(entry_stop=args.events)
        rechits_cluster_center = track_tree["rechitClusterCenter"].array(entry_stop=args.events)
        rechits_cluster_size = track_tree["rechitClusterSize"].array(entry_stop=args.events)
        digi_strip = track_tree["rechitDigiStrip"].array(entry_stop=args.events)
        raw_channel = track_tree["rechitRawChannel"].array(entry_stop=args.events)
        track_intercept_x = track_tree["trackInterceptX"].array(entry_stop=args.events)
        track_intercept_y = track_tree["trackInterceptY"].array(entry_stop=args.events)
        prophits_x = track_tree["prophitGlobalX"].array(entry_stop=args.events)
        prophits_y = track_tree["prophitGlobalY"].array(entry_stop=args.events)
        prophits_local_x = track_tree["prophitLocalX"].array(entry_stop=args.events)
        prophits_local_y = track_tree["prophitLocalY"].array(entry_stop=args.events)

        rechits_r = track_tree["rechitLocalR"].array(entry_stop=args.events)
        rechits_phi = track_tree["rechitLocalPhi"].array(entry_stop=args.events)
        prophits_r = track_tree["prophitLocalR"].array(entry_stop=args.events)
        prophits_phi = track_tree["prophitLocalPhi"].array(entry_stop=args.events)

        track_chi2 = (track_x_chi2*2 + track_y_chi2*2)/4
        mask_chi2 = (track_chi2>0.2)&(track_chi2<10)
        rechit_chamber = rechit_chamber[mask_chi2]
        prophit_chamber = prophit_chamber[mask_chi2]
        rechits_eta = rechits_eta[mask_chi2]
        rechits_cluster_center = rechits_cluster_center[mask_chi2]
        rechits_cluster_size = rechits_cluster_size[mask_chi2]
        digi_strip = digi_strip[mask_chi2]
        raw_channel = raw_channel[mask_chi2]
        rechits_x, rechits_y = rechits_x[mask_chi2], rechits_y[mask_chi2]
        rechits_local_x, rechits_local_y = rechits_local_x[mask_chi2], rechits_local_y[mask_chi2]
        prophits_x, prophits_y = prophits_x[mask_chi2], prophits_y[mask_chi2]
        prophits_local_x, prophits_local_y = prophits_local_x[mask_chi2], prophits_local_y[mask_chi2]
        track_intercept_x, track_intercept_y = track_intercept_x[mask_chi2], track_intercept_y[mask_chi2]
        track_x_chi2, track_y_chi2 = track_x_chi2[mask_chi2], track_y_chi2[mask_chi2]
        rechits_r, prophits_r = rechits_r[mask_chi2], prophits_r[mask_chi2]
        rechits_phi, prophits_phi = rechits_phi[mask_chi2], prophits_phi[mask_chi2]

        tested_chamber = args.chamber
        prophits_x, prophits_y = ak.flatten(prophits_x[prophit_chamber==tested_chamber]), ak.flatten(prophits_y[prophit_chamber==tested_chamber])
        prophits_local_x, prophits_local_y = ak.flatten(prophits_local_x[prophit_chamber==tested_chamber]), ak.flatten(prophits_local_y[prophit_chamber==tested_chamber])
        prophits_r, prophits_phi = ak.flatten(prophits_r[prophit_chamber==tested_chamber]), ak.flatten(prophits_phi[prophit_chamber==tested_chamber])
        rechits_x, rechits_y = rechits_x[rechit_chamber==tested_chamber], rechits_y[rechit_chamber==tested_chamber]
        rechits_local_x, rechits_local_y = rechits_local_x[rechit_chamber==tested_chamber], rechits_local_y[rechit_chamber==tested_chamber]
        rechits_r, rechits_phi = rechits_r[rechit_chamber==tested_chamber], rechits_phi[rechit_chamber==tested_chamber]
        rechits_eta = rechits_eta[rechit_chamber==tested_chamber]
        rechits_cluster_size = rechits_cluster_size[rechit_chamber==tested_chamber]
        digi_strip = digi_strip[rechit_chamber==tested_chamber]
        raw_channel = raw_channel[rechit_chamber==tested_chamber]
        rechits_cluster_center = rechits_cluster_center[rechit_chamber==tested_chamber]

        """ Choose only events within a (-30,30) mm window """
        prophit_window_mask = (abs(prophits_x)<30)&(abs(prophits_y)<30)
        prophits_x, prophits_y = prophits_x[prophit_window_mask], prophits_y[prophit_window_mask]
        prophits_local_x, prophits_local_y = prophits_local_x[prophit_window_mask], prophits_local_y[prophit_window_mask]
        prophits_r, prophits_phi = prophits_r[prophit_window_mask], prophits_phi[prophit_window_mask]

        rechits_x, rechits_y = rechits_x[prophit_window_mask], rechits_y[prophit_window_mask]
        rechits_local_x, rechits_local_y = rechits_local_x[prophit_window_mask], rechits_local_y[prophit_window_mask]
        rechits_r, rechits_phi = rechits_r[prophit_window_mask], rechits_phi[prophit_window_mask]
        rechits_eta, rechits_cluster_size = rechits_eta[prophit_window_mask], rechits_cluster_size[prophit_window_mask]
        digi_strip = digi_strip[prophit_window_mask]
        raw_channel = raw_channel[prophit_window_mask]
        rechits_cluster_center = rechits_cluster_center[prophit_window_mask]

        residuals_x, residuals_y = prophits_x-rechits_x, prophits_y-rechits_y
        residuals_r, residuals_phi = prophits_r-rechits_r, prophits_phi-rechits_phi
        residuals_local_x, residuals_local_y = prophits_local_x-rechits_local_x, prophits_local_y-rechits_local_y

        etas = np.unique(ak.flatten(rechits_eta))
        nrows, ncols = 2, len(etas)
        
       
        def residual_rotation(theta=0, x0=0, y0=0):

            os.makedirs(args.odir / str(theta), exist_ok=True)
            space_resolutions = list()

            n_eta = len(etas)
            strip_matching_fig, strip_matching_axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(11*ncols,9*nrows))
            residuals_x_fig, residuals_x_axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(11*ncols,9*nrows))
            strip_residuals_fig, strip_residuals_axs = plt.subplots(ncols=n_eta, nrows=1, figsize=(11*n_eta,9))
            residuals_fig, residuals_axs = plt.subplots(ncols=n_eta, nrows=1, figsize=(11*n_eta,9))
            residuals_phi_fig, residuals_phi_axs = plt.subplots(ncols=n_eta, nrows=1, figsize=(11*n_eta,9))
          
            print(etas)
            for ieta,eta in enumerate(etas):
                
                eta_mask = ak.any(rechits_eta==eta, axis=1)
                #eta_mask = ak.any(rechits_eta>0, axis=1)
                print("Rechits cluster size", rechits_cluster_size)
                eta_mask = eta_mask&(ak.all(rechits_cluster_size==2, axis=1))
                #eta_mask = ak.all(rechits_cluster_size==2, axis=1)

                print("eta mask", eta_mask)
                print("prophits x", prophits_x)
                print("digi strip", digi_strip)

                prophits_x_eta = prophits_x[eta_mask]
                prophits_y_eta = prophits_y[eta_mask]
                cls_eta = rechits_cluster_size[eta_mask]
                print("Cluster size after cut", cls_eta)
                digi_strip_eta = ak.mean(digi_strip[eta_mask], axis=1)

                """ Apply angular correction """
                prophits_x_corrected = (prophits_x_eta-x0) * np.cos(theta) - (prophits_y_eta-y0) * np.sin(theta)

                strip_window =  (digi_strip_eta>0)&(digi_strip_eta<385)
                prophits_x_corrected, prophits_y_eta, digi_strip_eta = prophits_x_corrected[strip_window], prophits_y_eta[strip_window], digi_strip_eta[strip_window]

                strip_matching_ax = strip_matching_axs[0][ieta]
                print(prophits_x_corrected, digi_strip_eta)
                strip_matching_ax.hist2d(
                    np.array(prophits_x_corrected), np.array(digi_strip_eta),
                    bins=100, range=((-30,30),(0,385))
                )
                strip_matching_ax.set_xlabel("Propagated x (mm)")
                strip_matching_ax.set_ylabel("Center strip")
                strip_matching_ax.set_title("$\eta$ = {}".format(eta))
            
                prophits_means, strip_edges, _ = scipy.stats.binned_statistic(digi_strip_eta, prophits_x_corrected, "mean", bins=50)
                prophits_std, strip_edges, _ = scipy.stats.binned_statistic(digi_strip_eta, prophits_x_corrected, "std", bins=50)
                prophits_count, strip_edges, _ = scipy.stats.binned_statistic(digi_strip_eta, prophits_x_corrected, "count", bins=50)

                strip_means = 0.5*(strip_edges[1:] + strip_edges[:-1])
                prophits_err = prophits_std# / np.sqrt(prophits_count)
                strip_err = 0.5*(strip_edges[1:] - strip_edges[:-1])
                strip_matching_ax = strip_matching_axs[1][ieta]
                strip_matching_ax.errorbar(
                    prophits_means, strip_means, yerr=prophits_err, xerr=strip_err, fmt="."
                )
                filter_nan = ~np.isnan(prophits_means)
                print(prophits_std[prophits_std==0])
                prophits_means, prophits_err, strip_means = prophits_means[filter_nan], prophits_err[filter_nan], strip_means[filter_nan]
            
                popt, pcov = curve_fit(linear_function, prophits_means, strip_means, p0=[300,1]) #,sigma=prophits_err
                print(eta, popt)
                to_strip = lambda x: linear_function(x, *popt)
                strip_matching_ax.plot(
                    prophits_means, to_strip(prophits_means),
                    color="red"
                )

                strip_matching_ax.set_ylim(0, 384)
                strip_matching_ax.set_xlabel("Propagated x (mm)")
                strip_matching_ax.set_ylabel("Center strip")
                strip_matching_ax.set_title("$\eta$ = {}".format(eta))
              
                
                """ Residuals in strip coordinates """ 

                propagated_strip = to_strip(prophits_x_corrected)
                residual_strip = propagated_strip - digi_strip_eta
               
                                
                """ Apply x correction """
                residuals_x_ax = residuals_x_axs[0][ieta]
                residuals_x_ax.hist2d(
                    prophits_x_corrected, residual_strip,
                    bins=100, #range=((-30,30),(-1, 1))
                )
                residuals_x_ax.set_xlabel("Propagated x (mm)")
                residuals_x_ax.set_ylabel("Residual (strips)")
                residuals_x_ax.set_title("$\eta$ = {}".format(eta))

                prophits_window = abs(prophits_x_corrected)<20
                prophits_x_corrected, residual_strip = prophits_x_corrected[prophits_window], residual_strip[prophits_window]
                residual_strip_means, x_edges, _ = scipy.stats.binned_statistic(prophits_x_corrected, residual_strip, "mean", bins=20)
                residual_strip_std, x_edges, _ = scipy.stats.binned_statistic(prophits_x_corrected, residual_strip, "std", bins=20)
                x_bins = 0.5 * (x_edges[1:] + x_edges[:-1])

                filter_nan = ~np.isnan(residual_strip_means)
                residual_strip_means, residual_strip_std, x_bins = residual_strip_means[filter_nan], residual_strip_std[filter_nan], x_bins[filter_nan]
                
                residuals_x_ax = residuals_x_axs[1][ieta]
                residuals_x_ax.errorbar(x_bins, residual_strip_means, yerr=residual_strip_std, fmt=".")
                x_opt, x_err = curve_fit(linear_function, x_bins, residual_strip_means, p0=[0, 0], sigma=residual_strip_std)
                print("x correction parameters", x_opt)
                residuals_x_ax.plot(x_bins, linear_function(x_bins, *x_opt), color="red")

                # correct for correlation with x:
                residual_strip = residual_strip - prophits_x_corrected * x_opt[1]
                
                """ Plot and fit residual distributions """ 
                residual_range, residual_bins = (-120,120), 80
                residual_binning = (residual_range[1]-residual_range[0])/residual_bins
                strip_residuals_ax = strip_residuals_axs[ieta]
                strip_hist, strip_edges, _ = strip_residuals_ax.hist(
                    residual_strip, bins=residual_bins, range=residual_range,
                    histtype="stepfilled", linewidth=2, facecolor="none", edgecolor="k",
                )
                strip_bins = 0.5 * (strip_edges[1:] + strip_edges[:-1])

                """ Gaussian fit of residual distribution """
                residuals_pars, residuals_cov = curve_fit(gauss, strip_bins, strip_hist, p0=[200, 0, 2])#, sigma=np.sqrt(strip_hist))
                residuals_err = np.sqrt(np.diag(residuals_cov)) 
                print("Residual parameters strips", residuals_pars, residuals_err)

                residual_sigma = residuals_pars[2]
                residual_sigma_err = residuals_err[2]

                space_resolutions.append(residual_sigma)

                strip_residuals_ax.text(
                        0.9, 0.85, f"$\sigma$ = {residual_sigma:1.2f} $\pm$ {residual_sigma_err:1.2f}",
                    transform = strip_residuals_ax.transAxes, ha="right"
                )
                strip_linspace = np.linspace(strip_bins[0], strip_bins[-1], 500)
                strip_residuals_ax.plot(strip_linspace, gauss(strip_linspace, *residuals_pars), color="red", linewidth=2)

                hep.cms.text(text="Preliminary", ax=strip_residuals_ax)
                strip_residuals_ax.set_xlabel("Residual strip")
                strip_residuals_ax.set_ylabel(f"Events / {residual_binning:1.1f} strips")
                strip_residuals_ax.text(
                    0.05, 0.9,
                    chamber_text,
                    transform=strip_residuals_ax.transAxes,
                    va="top", linespacing=1.7
                )
 

                """ Residuals in phi coordinates """ 
                #bmin, bmax, height, nstrips = 501.454, 659.804, 430.6, 384
                bmin, bmax, height, nstrips = 487.5, 631.5, 389.4, 384
                bmin, bmax, height, nstrips = 235.2, 460, 787.9, 384
                chamber_phi = 2 * np.arctan((bmax-bmin)/(2*height)) * 1e6
                pitch_phi = chamber_phi / nstrips
                print("Chamber phi", chamber_phi*1e-6*180/np.pi, "°")
                print("Pitch phi", pitch_phi, "µrad")
                residual_phi = residual_strip * pitch_phi
                print(residual_phi)

                residual_range, residual_bins = (-4500,4500), 80
                residual_binning = (residual_range[1]-residual_range[0])/residual_bins
                residuals_phi_ax = residuals_phi_axs[ieta]
                
                phi_hist, phi_edges, _ = residuals_phi_ax.hist(
                    residual_phi, bins=residual_bins, range=residual_range,
                    histtype="stepfilled", linewidth=2, facecolor="none", edgecolor="k",
                )
                phi_bins = 0.5 * (phi_edges[1:] + phi_edges[:-1])

                residuals_pars, residuals_cov = curve_fit(gauss, phi_bins, phi_hist, p0=[2000, 0, 350])#, sigma=np.sqrt(strip_hist))
                residuals_err = np.sqrt(np.diag(residuals_cov)) 
                print("Residual parameters phi", residuals_pars, residuals_err)

                hep.cms.text(text="Preliminary", ax=residuals_phi_ax)
                residuals_phi_ax.set_xlabel("Residual strip")
                residuals_phi_ax.set_ylabel(f"Events / {residual_binning:1.1f} strips")
                residuals_phi_ax.text(
                    0.05, 0.9,
                    chamber_text,
                    transform=residuals_phi_ax.transAxes,
                    va="top", linespacing=1.7
                )
                residuals_phi_ax.text(
                    1., 1., "H4 test beam",
                    va="bottom", ha="right",
                    transform = residuals_phi_ax.transAxes
                )


                residual_sigma = residuals_pars[2]
                residual_sigma_err = residuals_err[2]
                residuals_phi_ax.text(
                    0.95, 0.9, f"$\sigma$ = {residual_sigma:1.1f} $\pm$ {residual_sigma_err:1.1f} µrad",
                    transform = residuals_phi_ax.transAxes, ha="right", va="top"
                )
                phi_linspace = np.linspace(phi_bins[0], phi_bins[-1], 500)
                residuals_phi_ax.plot(phi_linspace, gauss(phi_linspace, *residuals_pars), color="red", linewidth=2)

                residuals_phi_ax.set_xlabel("Residual $\phi$ (µrad)")
                residuals_phi_ax.set_ylabel(f"Events / {residual_binning:1.1f} µrad")


                """ Residuals in x coordinates """ 
                residual_x = residual_strip * 1.5636

                residual_range, residual_bins = (-10,10), 50
                residual_binning = (residual_range[1]-residual_range[0])/residual_bins
                residuals_ax = residuals_axs[ieta]
                
                x_hist, x_edges, _ = residuals_ax.hist(
                    residual_x, bins=residual_bins, range=residual_range,
                    histtype="stepfilled", linewidth=2, facecolor="none", edgecolor="k",
                )
                x_bins = 0.5 * (x_edges[1:] + x_edges[:-1])

                residuals_pars, residuals_cov = curve_fit(gauss, x_bins, x_hist, p0=[200, 0, 2])#, sigma=np.sqrt(strip_hist))
                residuals_err = np.sqrt(np.diag(residuals_cov)) 
                print("Residual parameters x", residuals_pars, residuals_err)

                residual_sigma = residuals_pars[2]
                residual_sigma_err = residuals_err[2]
                residuals_ax.text(
                    0.9, 0.85, f"$\sigma$ = {residual_sigma:1.2f} $\pm$ {residual_sigma_err:1.2f}",
                    transform = residuals_ax.transAxes, ha="right"
                )
                residuals_ax.plot(x_bins, gauss(x_bins, *residuals_pars), color="red", linewidth=2)

                residuals_ax.set_xlabel("Residual x (mm)")
                residuals_ax.set_ylabel(f"Events / {residual_binning:1.1f} mm")

            strip_matching_fig.tight_layout()
            strip_matching_fig.savefig(args.odir/f"{theta}/prophits_strips.png")

            strip_residuals_fig.tight_layout()
            strip_residuals_fig.savefig(args.odir/f"{theta}/strip_residuals.png")

            residuals_fig.tight_layout()
            residuals_fig.savefig(args.odir/f"{theta}/residuals.png")

            residuals_x_fig.tight_layout()
            residuals_x_fig.savefig(args.odir/f"{theta}/residuals_y.png")

            residuals_phi_fig.tight_layout()
            residuals_phi_fig.savefig(args.odir/f"{theta}/residuals_phi.png")

            return space_resolutions

        resolutions = [list(), list()]
        rotation_angles = np.linspace(-5e-3, 5e-3, 5)
        #rotation_angles = np.linspace(-200, 200, 20)
        for correction in rotation_angles:
            print("Applying correction", correction)
            res = residual_rotation(theta=correction)#-100e-3, y0=correction, x0=correction)
            resolutions[0].append(res[0])
            resolutions[1].append(res[1])
        
        """ Plot space resolution vs rotation correction """ 
        rotation_fig, rotation_axs = plt.subplots(figsize=(24,9), nrows=1, ncols=2)
        for iax,ax in enumerate(rotation_axs):
            ax.plot(rotation_angles*1e3, resolutions[iax], ".")
            ax.set_xlabel("Rotation (mrad)")
            ax.set_ylabel("Residual sigma (strips)")
        rotation_fig.tight_layout()
        rotation_fig.savefig(args.odir / "rotation.png")

if __name__=='__main__': main()
