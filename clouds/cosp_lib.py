#!/usr/bin/env python3
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from numpy import ma
import xarray as xr
import cartopy.crs 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# author: Huan.Guo@noaa.gov
# compute global-mean 
def cal_gbl_mean (ds, var):
    weights = np.cos(np.deg2rad(ds['lat']))
    weights.name = "weights"
    gbl_mean_var = ds[var].weighted(weights).mean(dim=['lat', 'lon'])
    return  gbl_mean_var

def monthly_mean (ds, var):
    weights = np.cos(np.deg2rad(ds['lat']))
    weights.name = "weights"
    monthly_clim = ds[var].weighted(weights).mean(dim=['lat', 'lon']).groupby('time.month').mean(dim='time')
    return  monthly_clim
    
# author: Huan.Guo@noaa.gov
# compute bias between models and observations.
# var_mod (lon,lat)
def cal_bias (var_mod, var_obs, lon, lat):
  if var.shape == (len(lat), len(lon)):
     var = var.T  # transpose to (lon, lat)
  nlon = ma.shape(lon)[0]
  lonb = ma.zeros( [nlon+1], lon.dtype )
  lonb[0]    = 0.5*( lon[0] + lon[nlon-1] - 360.0 )
  lonb[nlon] = 0.5*( lon[0] + 360.0 + lon[nlon-1] )
  for i in range(1,nlon):
    lonb[i] = 0.5*( lon[i-1] + lon[i] )

  nlat = ma.shape(lat)[0]
  latb = ma.zeros( [nlat+1], lat.dtype )
  latb[0]      = -90.0
  latb[nlat]   =  90.0
  for i in range(1,nlat):
    latb[i] = 0.5*( lat[i-1] + lat[i] )

  latbr = latb / 180.0 * np.pi
  lonbr = lonb / 180.0 * np.pi 

  area = np.zeros( [nlat, nlon], 'float64' )
  for i in range(0,nlat):
    area[i,:] = (lonbr[i+1]-lonbr[i])*(np.sin(latbr[i+1])-np.sin(latbr[i]))
  var_tmp = var_mod - var_obs
  var_tmp_ma = np.ma.MaskedArray( var_tmp, mask=np.isnan(var_tmp))

  bias = np.ma.average( var_tmp_ma, weights=area )
  return bias

# author: Huan.Guo@noaa.gov
# RMSE, var_mod(lon,lat)
def cal_rmse (var_mod, var_obs, lon, lat):
  if var.shape == (len(lat), len(lon)):
     var = var.T
  nlon = ma.shape(lon)[0]
  lonb = ma.zeros( [nlon+1], lon.dtype )
  lonb[0]    = 0.5*( lon[0] + lon[nlon-1] - 360.0 )
  lonb[nlon] = 0.5*( lon[0] + 360.0 + lon[nlon-1] )
  for i in range(1,nlon):
    lonb[i] = 0.5*( lon[i-1] + lon[i] )

  nlat = ma.shape(lat)[0]
  latb = ma.zeros( [nlat+1], lat.dtype )
  latb[0]      = -90.0
  latb[nlat]   =  90.0
  for i in range(1,nlat):
    latb[i] = 0.5*( lat[i-1] + lat[i] )

  latbr = latb / 180.0 * np.pi
  lonbr = lonb / 180.0 * np.pi 

  area = np.zeros( [nlat, nlon], 'float64' )
  for i in range(0,nlat):
    area[i,:] = (lonbr[i+1]-lonbr[i])*(np.sin(latbr[i+1])-np.sin(latbr[i]))
  var_tmp = var_mod - var_obs
  var_tmp_ma = np.ma.MaskedArray( var_tmp, mask=np.isnan(var_tmp))

  rmse = np.sqrt( np.ma.average( var_tmp_ma*var_tmp_ma, weights=area ) )
  return  rmse

# author: Huan.Guo@noaa.gov
# corrleation, var_mod(lon,lat)
def cal_corr (var_mod, var_obs, lon, lat):
  if var.shape == (len(lat), len(lon)):
      var = var.T
  nlon = ma.shape(lon)[0]
  lonb = ma.zeros( [nlon+1], lon.dtype )
  lonb[0]    = 0.5*( lon[0] + lon[nlon-1] - 360.0 )
  lonb[nlon] = 0.5*( lon[0] + 360.0 + lon[nlon-1] )
  for i in range(1,nlon):
    lonb[i] = 0.5*( lon[i-1] + lon[i] )

  nlat = ma.shape(lat)[0]
  latb = ma.zeros( [nlat+1], lat.dtype )
  latb[0]      = -90.0
  latb[nlat]   =  90.0
  for i in range(1,nlat):
    latb[i] = 0.5*( lat[i-1] + lat[i] )

  latbr = latb / 180.0 * np.pi
  lonbr = lonb / 180.0 * np.pi 

  area = np.zeros( [nlat, nlon], 'float64' )
  for i in range(0,nlat):
    area[i,:] = (lonbr[i+1]-lonbr[i])*(np.sin(latbr[i+1])-np.sin(latbr[i]))

# there are NaN in observation and/or model outputs, 
# so save the mask of where NaN is,  
# then apply to both obs. and model results
  var_tmp = var_obs + var_mod
#  var_tmp = var_obs
  m = np.ma.MaskedArray( var_tmp, mask=np.isnan(var_tmp))

# calculate global average and std. dev. of observation
  var_tmp_ma = np.ma.MaskedArray( var_obs, mask=m.mask )
  var_obs_avg = np.ma.average( var_tmp_ma, weights=area ) 
  var_tmp =  var_obs - var_obs_avg
  var_tmp_ma = np.ma.MaskedArray( var_tmp, mask=m.mask )
  var_obs_std = np.sqrt( np.ma.average( var_tmp_ma*var_tmp_ma, weights=area ) )

# calculate global average and std. dev. of model results 
# using observation mask (missing values)
  var_tmp_ma = np.ma.MaskedArray( var_mod, mask=m.mask )
  var_mod_avg = np.ma.average( var_tmp_ma, weights=area )
  print(  'var_mod_avg = ',  var_mod_avg)
  var_tmp =  var_mod - var_mod_avg 
  var_tmp_ma = np.ma.MaskedArray( var_tmp, mask=m.mask )
  var_mod_std = np.sqrt( np.ma.average( var_tmp_ma*var_tmp_ma, weights=area ) ) 

  var_tmp = (var_mod-var_mod_avg)*(var_obs-var_obs_avg)
  var_tmp_ma = np.ma.MaskedArray( var_tmp, mask=m.mask )
  corr = np.ma.average( var_tmp_ma, weights=area )/(var_obs_std * var_mod_std   )
  return  corr

#---- plot MODIS cloud fraction map ------------------------------------------------------------
def plot_modis_cldfrac (var_mod,lon, lat):
    fig = plt.figure(figsize=(7.0, 13.0))
    fig.subplots_adjust( top=0.99, bottom=0.02, left=0.07, right=0.991, hspace=0.0, wspace=0.0 )
    ax11 = plt.subplot(311, projection=cartopy.crs.Robinson(central_longitude=-180) )
    ax11.set_facecolor('xkcd:gray')
    projection = cartopy.crs.PlateCarree()
    im11 = ax11.contourf(lon,lat, var_obs, cmap= 'jet', levels=np.linspace(0, 101.0, 9), vmax=100,vmin=0.0, transform=projection,extend='both')
    ax11.gridlines(linewidth=1.3);
    ax11.text(0.6, 1.2, 'MODIS', fontsize=35,  horizontalalignment='right', verticalalignment='center', transform=ax11.transAxes) 
    #ax11.set_title('(a) Avg=%.1f'%gbl_mean_MODIS_total_cld, fontsize=31, fontweight ='normal')
    ax11.set_title('(a) %.1f'%cal_gbl_mean(var_obs, lon, lat), fontsize=31, fontweight ='normal')
    ax11.text(-0.04, 0.5, 'Obs.', fontsize=32, horizontalalignment='right', verticalalignment='center', transform=ax11.transAxes)
    ax11.set_global()
    ax11.coastlines()