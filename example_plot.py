#example residual plot

# this example compute the residual against different CMB (and nuisance) best fits and display them

import pylab as plt
import numpy as nm # see how old I am... Back then, Numeric and then numarray were often abbreviated as nm

# contains all the helper functions to compute coadded and play with nuisance parameters values
import clik.smicahlp as smh

# simple function to create binninb matrices
def create_binning(ls,delta=None,wgh=None,lmin=-1,lmax=-1,norm=True):
  if len(ls)==2:
    _ls = nm.arange(ls[0],ls[1]+1,delta)
    if _ls[-1]!=ls[1]+1:
      _ls = nm.concatenate((_ls,[ls[1]+1]))
    ls = _ls
  if wgh is None:
    wgh = nm.ones((ls[-1]-ls[0]))
  assert len(wgh) == ls[-1]-ls[0]
  if lmin==-1:
    lmin=ls[0]
  if lmax==-1:
    lmax=ls[-1]-1
  bns = nm.zeros((len(ls)-1,lmax+1-lmin))
  for i in range(len(ls)-1):
    bns[i,ls[i]-lmin:ls[i+1]-lmin]=wgh[ls[i]-ls[0]:ls[i+1]-ls[0]]
    if norm:
      bns[i,ls[i]-lmin:ls[i+1]-lmin] /= nm.sum(bns[i,ls[i]-lmin:ls[i+1]-lmin])
  return bns

# helper for plots
def cm2inch(cm):
    """Centimeters to inches"""
    return cm *0.393701

# helper for plots, add a text on a curve
import matplotlib.patheffects as PathEffects
def add_curvelabel(txt,lin,posx=None,size=9,color="",**extra):
  x = lin.get_xdata()
  y = lin.get_ydata()
  if posx==None:
    posx = nm.mean(x)
  posy = nm.interp(posx,x,y)
  ax = lin.axes
  pn = ax.transData.transform(list(zip(x,y)))
  pn0 = ax.transData.transform([posx,posx])
  der = nm.interp(pn0[0],pn[:-1,0] + nm.diff(pn[:,0])/2.,nm.diff(pn[:,1])/nm.diff(pn[:,0]))
  rot = nm.rad2deg(nm.arctan(der))
  if not color:
    color = lin.get_color()
  ttx = plt.text(posx, posy, txt, size=size, rotation=rot, color = color,
     ha="center", va="center",path_effects=[PathEffects.withStroke(linewidth=3,foreground="w")],**extra) #,bbox = dict(ec='1',fc='1')
  def updaterot(ax):
    pn = ax.transData.transform(list(zip(x,y)))
    pn0 = ax.transData.transform([posx,posx])
    der = nm.interp(pn0[0],pn[:-1,0] + nm.diff(pn[:,0])/2.,nm.diff(pn[:,1])/nm.diff(pn[:,0]))
    rot = nm.rad2deg(nm.arctan(der))
    #print rot
    ttx.set_rotation(rot)
  ax.callbacks.connect("xlim_changed",updaterot)
  ax.callbacks.connect("ylim_changed",updaterot)


# where is the clikfile
lkl_TT_bin1 = "data/clikfiles/plik_rd12_HM_v22_TT_bin1.clik/"

# where are the best fit data
# this assumes that the best fit data is given as a cosmomc best fit. It needs (in that order) the best fit CL (in Dl, cosmomc format), the best fit parameters and the cosmomc paramnames file
# this can be replaced by a single text file that contains the cl values from 0 to lmax (in CL !!) followed by the nuisance parameter best fit values in the order expected by the likelihood file

# best fit hi-l TT only
bf_TT = ("data/bestfits/base_plikHM_TT_lowl_lowE.minimum.theory_cl","data/bestfits/base_plikHM_TT_lowl_lowE.minimum","data/paramnames/plik_rd12_HM_v22_TT.paramnames")

# best fit TT only with Alens
bf_TT_Alens_TTonly = ("data/bestfits/base_Alens_plik_rd12_HM_v22_TT_lowlv3_simall_EE.minimum.theory_cl","data/bestfits/base_Alens_plik_rd12_HM_v22_TT_lowlv3_simall_EE.minimum","data/paramnames/plik_rd12_HM_v22_TT.paramnames")

# this function computes the coadded best CMB for a given clikfile (here lkl_TT_bin1) and a given best fit theory cl and nuisance parameters (in this case, the one defined by bf_TT)
tm, tVec,eVec,Jt_siginv_J,good = smh.best_fit_cmb(lkl_TT_bin1,bf_TT,cal=True,rcal=False,goodmask=True)
# tm contains the (possibly binned) l list where the coadded in defined. It is a 2D array, first coordinates correspond to choice of TT, EE, BB, TE, TE, EB
# tVec it the coadded Cl (again 2D array)
# Jt_siginv_J is the inverse covariance of the coadded Cl
# the rest are specific output for more complicated stuff !

# now do the same for the Alens TT only case
tm, tVec2TTonly,eVec2TTonly,Jt_siginv_J2TTonly,good2TTonly = smh.best_fit_cmb(lkl_TT_bin1,bf_TT_Alens_TTonly,cal=True,rcal=False,goodmask=True)

# get the covariance of the coadded Cl
sig = nm.linalg.inv(Jt_siginv_J)
# note that the coadded cov does not depend on the nuisance values !

# compute llp1 for the ells we use
llp1 = tm[0]*(tm[0]+1)/2./nm.pi

# define binning matrix at deltal=50
deltabin=50
bns = create_binning((0,len(tm[0])-1),delta = deltabin)
# compute bin locations and approximate binned llp1
lmb = nm.dot(bns,tm[0])
llp1_b = lmb*(lmb+1)/2./nm.pi

# compute binned errorbars and multiply by approximated binned llp1
errb = nm.sqrt(nm.dot(bns,nm.dot(sig,bns.T)).diagonal())*llp1_b


# retrive the best fit nuisance parameters and the best fit theory cl
bfl = smh.get_bestfit_and_cl(lkl_TT_bin1,bf_TT)
# bfl is a tuple, first element is a dict containing the best fit nuisance parameter values, second element are the best fit Cl

# do the same for different best fit cosmologies
bfl_Alens_TTonly = smh.get_bestfit_and_cl(lkl_TT_bin1,bf_TT_Alens_TTonly)


# compute residual between coadded using bf_TT nuisance parameters and the bf_TT best fit
delta_b = nm.dot(bns,(tVec[0]-bfl[1][0,30:]))


# compute residual between coadded using bf_TT_Alens_TTonly nuisance parameters and the bf_TT best fit
delta_b2TTonly = nm.dot(bns,(tVec2TTonly[0]-bfl[1][0,30:])) 

# start plotting
plt.figure(figsize=(cm2inch(8.8)*1.5,cm2inch(8.8)*1.5/2.))

# the residuals
l1TTonly=plt.errorbar(lmb,delta_b2TTonly*llp1_b,errb,lw=0.3,fmt='o',c=plt.cm.tab20(2),mew=0,ms=3,capsize=0,label="Coadded ($A_\mathrm{L}$ foregrounds)")

# shift a bit the center of the plots so that the residuals are not on top of each other !
l1b=plt.errorbar(lmb+15,delta_b*llp1_b,errb,lw=0.3,fmt='o',c=plt.cm.tab20(4),mew=0,ms=3,capsize=0,label="Coadded ($\Lambda {\\rm CDM}$ foregrounds)",alpha=.9)

# the comparison between the lambda cdm and Alens best fits
l2b,=plt.plot(tm[0],bfl_Alens_TTonly[1][0,30:]*llp1-bfl[1][0,30:]*llp1,label="$\mathcal{D}^{A_\mathrm{L}}_\ell-\mathcal{D}^{\Lambda {\\rm CDM}}_\ell(TT)$",c=plt.cm.tab20(0))

# pretty plot CMB/100 to look at oscillations
l,=plt.plot(tm[0],bfl[1][0,30:]*llp1/100,label="$1\\%$ CMB",lw=.7,c="grey",alpha=.5)
add_curvelabel("$1\%$ CMB",l,750,size=4)

# finish plot and beautify
plt.axhline(0,c="k",lw=.5)
plt.ylim(-40,40)
plt.legend(handles=[l1TTonly,l1b,l2b],frameon=False,fontsize=5,ncol=2,loc="upper right")
plt.ylabel("$\Delta\mathcal{D}_\ell\  [\mu  \mathrm {K}^2]$")
plt.tight_layout()
plt.savefig("My_Alens_plot.pdf")
