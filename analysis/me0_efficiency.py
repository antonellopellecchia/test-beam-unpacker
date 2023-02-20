import ROOT


class rechit():
    def __init__(self, x, y, chamber, eta, vfat, strip):
        self.x = x
        self.y = y
        self.chamber = chamber
        self.eta = eta
        self.vfat = vfat
        self.strip = strip

def residual_match(hits, chamber):
    if chamber == 1: return (2*(hits[1].x) - (hits[2].x)) - hits[0].x
    elif chamber == 2: return ((hits[0].x + hits[2].x)/2.0) - hits[1].x
    elif chamber == 3: return (2*(hits[1].x) - (hits[0].x)) - hits[2].x
    else: return 999

debug = False

H_ref = 800
W_ref = 800
W = W_ref
H = H_ref

T = 0.12*H_ref
B = 0.16*H_ref
L = 0.16*W_ref
R = 0.08*W_ref

fname = "run_10022023_1700_rechits_17feb.root"
f = ROOT.TFile(fname)
events = f.Get("rechitTree")

nHitsMax = 1
plot_prefix = "plt_nHit{}_".format(nHitsMax)

hist_chamber1_nHits = ROOT.TH1F("hist_chamber1_nHits", "hist_chamber1_nHits", 200, 0, 200)
hist_chamber2_nHits = ROOT.TH1F("hist_chamber2_nHits", "hist_chamber2_nHits", 200, 0, 200)
hist_chamber3_nHits = ROOT.TH1F("hist_chamber3_nHits", "hist_chamber3_nHits", 200, 0, 200)

hist_residual_chamber1_eta_list = []
hist_residual_chamber2_eta_list = []
hist_residual_chamber3_eta_list = []

eff_by_strip_chamber1_eta_list = []
eff_by_strip_chamber2_eta_list = []
eff_by_strip_chamber3_eta_list = []
nStrips = 128*3
nBins = 12

eff_chamber1 = ROOT.TEfficiency("eff_chamber1", "eff_chamber1", 4, 5, 9)
eff_chamber2 = ROOT.TEfficiency("eff_chamber2", "eff_chamber2", 4, 5, 9)
eff_chamber3 = ROOT.TEfficiency("eff_chamber3", "eff_chamber3", 4, 5, 9)

for eta in range(1,9):
    hist_residual_chamber1_eta_list.append(ROOT.TH1F("hist_residual_chamber1_eta{}".format(eta), "hist_residual_chamber1_eta{}".format(eta), 20, -5, 5))
    hist_residual_chamber2_eta_list.append(ROOT.TH1F("hist_residual_chamber2_eta{}".format(eta), "hist_residual_chamber2_eta{}".format(eta), 20, -5, 5))
    hist_residual_chamber3_eta_list.append(ROOT.TH1F("hist_residual_chamber3_eta{}".format(eta), "hist_residual_chamber3_eta{}".format(eta), 20, -5, 5))

    eff_by_strip_chamber1_eta_list.append(ROOT.TEfficiency("eff_chamber1_eta{}".format(eta), "eff_chamber1_eta{}".format(eta), nBins, 0, nStrips))
    eff_by_strip_chamber2_eta_list.append(ROOT.TEfficiency("eff_chamber2_eta{}".format(eta), "eff_chamber2_eta{}".format(eta), nBins, 0, nStrips))
    eff_by_strip_chamber3_eta_list.append(ROOT.TEfficiency("eff_chamber3_eta{}".format(eta), "eff_chamber3_eta{}".format(eta), nBins, 0, nStrips))

nEvents = events.GetEntries()
print("There are ", nEvents, " events")
for index, event in enumerate(events):
    if (index%(int(nEvents/10))) == 0: print("At event ", event.eventCounter, " ", int(index/nEvents*100))


    ch1_list = []
    ch2_list = []
    ch3_list = []
    for i in range(event.nrechits):
        if event.rechitChamber[i] == 1:
            ch1_list.append(rechit(event.rechitX[i], event.rechitY[i], event.rechitChamber[i], event.rechitEta[i], event.rechitVFAT[i], event.rechitStrip[i]))
        if event.rechitChamber[i] == 2:
            ch2_list.append(rechit(event.rechitX[i], event.rechitY[i], event.rechitChamber[i], event.rechitEta[i], event.rechitVFAT[i], event.rechitStrip[i]))
        if event.rechitChamber[i] == 3:
            ch3_list.append(rechit(event.rechitX[i], event.rechitY[i], event.rechitChamber[i], event.rechitEta[i], event.rechitVFAT[i], event.rechitStrip[i]))

    if (len(ch1_list) > 50 or len(ch2_list) > 50 or len(ch3_list) > 50) and debug:
        print("New event! ", event.eventCounter)
        print("Hit lists = ")
        print("Len of ch1 = ", len(ch1_list))
        print("Len of ch2 = ", len(ch2_list))
        print("Len of ch3 = ", len(ch3_list))

    hist_chamber1_nHits.Fill(len(ch1_list))
    hist_chamber2_nHits.Fill(len(ch2_list))
    hist_chamber3_nHits.Fill(len(ch3_list))


    nHitsMax = 1
    if not (len(ch1_list) <= nHitsMax and len(ch2_list) <= nHitsMax and len(ch3_list) <= nHitsMax): continue
    ch3_match = None
    ch3_res = 999
    for i, ch1_hit in enumerate(ch1_list):
        for j, ch2_hit in enumerate(ch2_list):
            ch3_prop = (2*ch2_hit.x - ch1_hit.x)
            for k, ch3_hit in enumerate(ch3_list):
                if abs(ch3_prop - ch3_hit.x) < abs(ch3_res):
                    ch3_match = [ch1_hit, ch2_hit, ch3_hit]
            if ch3_match == None:
                dummy_rechit = rechit(999, 999, 3, ch2_hit.eta, ch2_hit.vfat, 2*ch2_hit.strip - ch1_hit.strip)
                ch3_match = [ch1_hit, ch2_hit, dummy_rechit]


    ch2_match = None
    ch2_res = 999
    for i, ch1_hit in enumerate(ch1_list):
        for j, ch3_hit in enumerate(ch3_list):
            ch2_prop = (2*ch2_hit.x - ch1_hit.x)
            for k, ch2_hit in enumerate(ch2_list):
                if abs(((ch1_hit.x + ch3_hit.x)/2.0)) < abs(ch2_res):
                    ch2_match = [ch1_hit, ch2_hit, ch3_hit]
            if ch2_match == None:
                dummy_rechit = rechit(999, 999, 2, int((ch1_hit.eta+ch3_hit.eta)/2), int((ch1_hit.vfat+ch3_hit.vfat)/2), int((ch1_hit.strip+ch3_hit.strip)/2))
                ch2_match = [ch1_hit, dummy_rechit, ch3_hit]


    ch1_match = None
    ch1_res = 999
    for j, ch2_hit in enumerate(ch2_list):
        for k, ch3_hit in enumerate(ch3_list):
            ch1_prop = (2*ch3_hit.x - ch2_hit.x)
            for i, ch1_hit in enumerate(ch1_list):
                if abs(ch1_prop - ch1_hit.x) < abs(ch1_res):
                    ch1_match = [ch1_hit, ch2_hit, ch3_hit]
            if ch1_match == None:
                dummy_rechit = rechit(999, 999, 1, ch2_hit.eta, ch2_hit.vfat, 2*ch2_hit.strip - ch3_hit.strip)
                ch1_match = [dummy_rechit, ch2_hit, ch3_hit]


    if (len(ch2_list) != 0 and len(ch3_list) != 0):
        hist_residual_chamber1_eta_list[ch1_match[0].eta-1].Fill(residual_match(ch1_match,1))
        eff_by_strip_chamber1_eta_list[ch1_match[0].eta-1].Fill(residual_match(ch1_match,1) < 5, ch1_match[0].strip)
        eff_chamber1.Fill(residual_match(ch1_match,1) < 5, ch1_match[0].eta)
    if (len(ch1_list) != 0 and len(ch3_list) != 0):
        hist_residual_chamber2_eta_list[ch2_match[1].eta-1].Fill(residual_match(ch2_match,2))
        eff_by_strip_chamber2_eta_list[ch2_match[1].eta-1].Fill(residual_match(ch2_match,2) < 5, ch2_match[1].strip)
        eff_chamber2.Fill(residual_match(ch2_match,2) < 5, ch2_match[1].eta)
    if (len(ch1_list) != 0 and len(ch2_list) != 0):
        hist_residual_chamber3_eta_list[ch3_match[2].eta-1].Fill(residual_match(ch3_match,3))
        eff_by_strip_chamber3_eta_list[ch3_match[2].eta-1].Fill(residual_match(ch3_match,3) < 5, ch3_match[2].strip)
        eff_chamber3.Fill(residual_match(ch3_match,3) < 5, ch3_match[2].eta)






canvas1 = ROOT.TCanvas("c1", "c1", 100, 100, W, H)
canvas1.SetFillColor(0)
canvas1.SetBorderMode(0)
canvas1.SetFrameFillStyle(0)
canvas1.SetFrameBorderMode(0)
canvas1.SetLeftMargin( L/W )
canvas1.SetRightMargin( R/W )
canvas1.SetTopMargin( T/H )
canvas1.SetBottomMargin( B/H )
canvas1.SetTickx(0)
canvas1.SetTicky(0)

canvas1.Divide(1,3)
p1 = canvas1.cd(1)
hist_chamber1_nHits.Draw()
xaxis = hist_chamber1_nHits.GetXaxis()
xaxis.SetTitle("nHits")
yaxis = hist_chamber1_nHits.GetYaxis()
yaxis.SetTitle("Entries")
p1.SetLogy()
p2 = canvas1.cd(2)
hist_chamber2_nHits.Draw()
xaxis = hist_chamber2_nHits.GetXaxis()
xaxis.SetTitle("nHits")
yaxis = hist_chamber2_nHits.GetYaxis()
yaxis.SetTitle("Entries")
p2.SetLogy()
p3 = canvas1.cd(3)
hist_chamber3_nHits.Draw()
xaxis = hist_chamber3_nHits.GetXaxis()
xaxis.SetTitle("nHits")
yaxis = hist_chamber3_nHits.GetYaxis()
yaxis.SetTitle("Entries")
p3.SetLogy()

canvas1.SaveAs(plot_prefix+"nHits_by_chamber.png")



canvas2 = ROOT.TCanvas("c2", "c2", 100, 100, W, H)
canvas2.SetFillColor(0)
canvas2.SetBorderMode(0)
canvas2.SetFrameFillStyle(0)
canvas2.SetFrameBorderMode(0)
canvas2.SetLeftMargin( L/W )
canvas2.SetRightMargin( R/W )
canvas2.SetTopMargin( T/H )
canvas2.SetBottomMargin( B/H )
canvas2.SetTickx(0)
canvas2.SetTicky(0)

canvas2.Divide(4,3)
for eta in range(5, 9):
    canvas2.cd(eta-4)
    hist_residual_chamber1_eta_list[eta-1].Draw()
    xaxis = hist_residual_chamber1_eta_list[eta-1].GetXaxis()
    xaxis.SetTitle("Residual [mm]")
    yaxis = hist_residual_chamber1_eta_list[eta-1].GetYaxis()
    yaxis.SetTitle("Entries")
    canvas2.cd(eta-4+4)
    hist_residual_chamber2_eta_list[eta-1].Draw()
    xaxis = hist_residual_chamber2_eta_list[eta-1].GetXaxis()
    xaxis.SetTitle("Residual [mm]")
    yaxis = hist_residual_chamber2_eta_list[eta-1].GetYaxis()
    yaxis.SetTitle("Entries")
    canvas2.cd(eta-4+8)
    hist_residual_chamber3_eta_list[eta-1].Draw()
    xaxis = hist_residual_chamber3_eta_list[eta-1].GetXaxis()
    xaxis.SetTitle("Residual [mm]")
    yaxis = hist_residual_chamber3_eta_list[eta-1].GetYaxis()
    yaxis.SetTitle("Entries")

canvas2.SaveAs(plot_prefix+"residual_by_eta.png")


canvas3 = ROOT.TCanvas("c3", "c3", 100, 100, W, H)
canvas3.SetFillColor(0)
canvas3.SetBorderMode(0)
canvas3.SetFrameFillStyle(0)
canvas3.SetFrameBorderMode(0)
canvas3.SetLeftMargin( L/W )
canvas3.SetRightMargin( R/W )
canvas3.SetTopMargin( T/H )
canvas3.SetBottomMargin( B/H )
canvas3.SetTickx(0)
canvas3.SetTicky(0)


canvas3.Divide(4,3)
for eta in range(5, 9):
    canvas3.cd(eta-4)
    gr1 = eff_by_strip_chamber1_eta_list[eta-1].CreateGraph()
    gr1.Draw("ap")
    yaxis = gr1.GetYaxis()
    yaxis.SetRangeUser(0.0, 1.0)
    yaxis.SetTitle("Efficiency")
    xaxis = gr1.GetXaxis()
    xaxis.SetTitle("Strips")
    canvas3.cd(eta-4+4)
    gr2 = eff_by_strip_chamber2_eta_list[eta-1].CreateGraph()
    gr2.Draw("ap")
    yaxis = gr2.GetYaxis()
    yaxis.SetRangeUser(0.0, 1.0)
    yaxis.SetTitle("Efficiency")
    xaxis = gr2.GetXaxis()
    xaxis.SetTitle("Strips")
    canvas3.cd(eta-4+8)
    gr3 = eff_by_strip_chamber3_eta_list[eta-1].CreateGraph()
    gr3.Draw("ap")
    yaxis = gr3.GetYaxis()
    yaxis.SetRangeUser(0.0, 1.0)
    yaxis.SetTitle("Efficiency")
    xaxis = gr3.GetXaxis()
    xaxis.SetTitle("Strips")


canvas3.SaveAs(plot_prefix+"eff_by_eta.png")






canvas4 = ROOT.TCanvas("c4", "c4", 100, 100, 3*W, H)
canvas4.SetFillColor(0)
canvas4.SetBorderMode(0)
canvas4.SetFrameFillStyle(0)
canvas4.SetFrameBorderMode(0)
canvas4.SetLeftMargin( L/W )
canvas4.SetRightMargin( R/W )
canvas4.SetTopMargin( T/H )
canvas4.SetBottomMargin( B/H )
canvas4.SetTickx(0)
canvas4.SetTicky(0)

canvas4.Divide(3)
p1 = canvas4.cd(1)
gr1 = eff_chamber1.CreateGraph()
gr1.Draw("ap")
xaxis = gr1.GetXaxis()
xaxis.SetTitle("Eta Partition")
yaxis = gr1.GetYaxis()
yaxis.SetTitle("Efficiency")
yaxis.SetRangeUser(0.0, 1.0)
p2 = canvas4.cd(2)
gr2 = eff_chamber2.CreateGraph()
gr2.Draw("ap")
xaxis = gr2.GetXaxis()
xaxis.SetTitle("Eta Partition")
yaxis = gr2.GetYaxis()
yaxis.SetTitle("Efficiency")
yaxis.SetRangeUser(0.0, 1.0)
p3 = canvas4.cd(3)
gr3 = eff_chamber3.CreateGraph()
gr3.Draw("ap")
xaxis = gr3.GetXaxis()
xaxis.SetTitle("Eta Partition")
yaxis = gr3.GetYaxis()
yaxis.SetTitle("Efficiency")
yaxis.SetRangeUser(0.0, 1.0)

canvas4.SaveAs(plot_prefix+"eff.png")




canvas5 = ROOT.TCanvas("c5", "c5", 100, 100, 3*W, H)
canvas5.Divide(3)
rechits_hlist = []
for i in range(1, 4):
    canvas5.cd(i)
    rechits_hlist.append(ROOT.TH2F("hist_hits_layer{i}".format(i = i), "hist_hits_layer{i}".format(i = i), 50, -300, 300, 8, 1, 9))
    events.Project("hist_hits_layer{i}".format(i = i), "rechitEta:rechitX", "rechitChamber == {i}".format(i = i))

    rechits_hlist[i-1].Draw("colz")
    xaxis = rechits_hlist[i-1].GetXaxis()
    yaxis = rechits_hlist[i-1].GetYaxis()
    xaxis.SetTitle("x [mm]")
    yaxis.SetTitle("Eta Partition")

canvas5.SaveAs(plot_prefix+"2D_hits.png")
