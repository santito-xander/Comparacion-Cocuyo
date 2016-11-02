import matplotlib 
#matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg
import matplotlib.cm as cm
from   matplotlib.colorbar import ColorbarBase
from   matplotlib.colors import Normalize
from   matplotlib.pyplot import figure
from   scipy.interpolate import griddata
import matplotlib.mlab as mlab
import pandas as pd
import numpy as np 
import math
import os 

#############################################################################
def interpolaDatos(THETA,R,magnitude):
  
  THETAi  = np.linspace(min(THETA),max(THETA),num=720.0) #720.0
  
  Ri      = np.linspace(min(R),max(R),num=360.0) #360.0
  
  theta,r = np.meshgrid(THETAi,Ri,indexing='xy') 
  
  z = griddata((THETA,R),magnitude,(theta,r),method="linear")
  
  z = np.round(z,2)
  
  return((THETAi,Ri,z))
  
def fixMagnitude(magnitude):
  
  inx0    = np.arange(0,6) #0,6
  
  inx360  = np.arange(len(magnitude) - 6,len(magnitude))
  
  magnitude0    = magnitude[inx0]
  
  magnitude360  = magnitude[inx360]
  
  magnitudePromedio = (magnitude0 + magnitude360)/2
  
  magnitude[inx0]   = magnitudePromedio 
  
  magnitude[inx360] = magnitudePromedio    

  return(magnitude)
  
def extraeColumnas(df):
  
  azimuth   = np.array(df['azimuth'])
  elevation = np.array(df['elevation'])
  magnitude = np.array(df['magnitude'])
 
  return((azimuth,elevation,magnitude))
  
def seleccionaColor(miColor):
  if (miColor == "azul"):
    micolor = cm.YlGnBu
  elif (miColor == "anaranjado"):
    micolor = cm.autumn
  elif (miColor == "gris"):
    micolor = cm.Greys 
  return(micolor)  
  
#############################################################################
def dosGraficas(data,
                data2,
                miTitulo,
                miTitulo2,
                miFecha,
                miFecha2,
                miColor="azul",
                puntosMuestreo=False,
                direccionReloj=False,
                mostrarContorno=False):
    #Extrae los datos del Data frame para la primera grafica
   
    datos = pd.DataFrame(data,columns=['azimuth','elevation','magnitude'])
    azimuth,elevation,magnitude = extraeColumnas(datos)
    theta = np.radians(azimuth)
    phi   = np.radians(elevation)
    magnitude = fixMagnitude(magnitude)
    thetai,phii,z = interpolaDatos(theta,phi,magnitude)
    
    #Extrae los datos del Data frame para la segunda grafica
    
   
    datos2 = pd.DataFrame(data2,columns=['azimuth','elevation','magnitude']) 
    azimuth2,elevation2,magnitude2 = extraeColumnas(datos2)
    theta2 = np.radians(azimuth2)
    phi2   = np.radians(elevation2)
    magnitude2 = fixMagnitude(magnitude2)
    thetai2,phii2,z2 = interpolaDatos(theta2,phi2,magnitude2)
    
    
#############################################################################
    #codigo compratido
    micolor = seleccionaColor(miColor)
    fig = figure(figsize=(11.0,5.9)) #(8.0,8.6) 
    vmin = 18.0 #18.0
    vmax = 22.0 #22.0
    cax = fig.add_axes([0.33, 0.05, 0.35, 0.04]) #[0.25,0.03,0.5,0.03]
    
    
    
#############################################################################
    # primera grafica
    ax = pyplot.subplot(121, projection='polar')
    
    ax.contourf(thetai,phii,z,10,linewidth=2,cmap=micolor,vmin=vmin,vmax=vmax)
    if direccionReloj:
       ang = 0.0
    else:
       ang = math.pi/2
       
    labels = np.unique(theta)
      
    ax.set_xticks(labels[:-1])
    ax.set_yticks(())
    
    ax.grid(True)
    
    ax.text(ang,1.6,miTitulo,fontsize=18,horizontalalignment='center')
    ax.text(ang,1.4,miFecha,fontsize=18,horizontalalignment='center')
    
    ax.set_xticklabels(['N', '$30^o$', '$60^o$', 'E', 
    '$120^o$', '$150^o$','S', '$210^o$', '$240^o$','W', '$300^o$','$330^o$'],
    ha='center') 
                  
    if mostrarContorno:
       ax.contour(thetai,phii,z,10,colors="k") #(10)
       
    if puntosMuestreo:
       ax.scatter(theta,phi)
       ax.set_rmin(0)
       ax.set_rmax(1.2) #(1.2)
       
    if direccionReloj:  
       ax.set_theta_direction(-1) #(-1)
       ax.set_theta_offset(math.pi/2.0) #(math.pi/2.0) 
       
    if direccionReloj:
       ax.text(np.radians(151),1.7,"$mag/arcsec^2$",
       weight="bold",
       fontsize=16)
    else:
       ax.text(np.radians(299),1.7,"$mag/arcsec^2$",
            weight="bold",
            fontsize=16)   
       
#############################################################################
    # segunda grafica 
    ax = pyplot.subplot(122, projection='polar')
    
    ax.contourf(thetai2,phii2,z2,10,linewidth=2,cmap=micolor,vmin=vmin,vmax=vmax)
    if direccionReloj:
       ang = 0.0
    else:
       ang = math.pi/2  
    
    ax.grid(True)
    
    labels = np.unique(theta)
    ax.set_xticks(labels[:-1])

    ax.set_yticks(())
    
    ax.text(ang,1.6,miTitulo,fontsize=18,horizontalalignment='center')
    ax.text(ang,1.4,miFecha,fontsize=18,horizontalalignment='center')
    
    ax.set_xticklabels(['N', '$30^o$', '$60^o$', 'E', 
    '$120^o$', '$150^o$','S', '$210^o$', '$240^o$','W', '$300^o$','$330^o$'],
    ha='center') 
    
    if mostrarContorno:
      ax.contour(thetai2,phii2,z2,10,colors="k") #(10)
      
    if puntosMuestreo:
       ax.scatter(theta,phi)
       ax.set_rmin(0)
       ax.set_rmax(1.2) #(1.2)
    
    if direccionReloj:  
       ax.set_theta_direction(-1) #(-1)
       ax.set_theta_offset(math.pi/2.0) #(math.pi/2.0)
       
    ColorbarBase(cax,orientation='horizontal',cmap=micolor,
               spacing='proportional',
               norm=Normalize(vmin=vmin, vmax=vmax))    
  
    return( pyplot.show())
#############################################################################    

data = np.loadtxt("ejemplo.dat")
data2 = np.loadtxt("ejemplo2.dat")

outfile   = "ejemplo.png"
miTitulo  = "ejemplo.png"
miFecha   = "00/00/0000"
miColor   = "azul"

outfile2   = "ejemplo2.png"
miTitulo2  = "ejemplo2.png"
miFecha2   = "00/00/0000"

puntosMuestreo  = True
direccionReloj  = True
mostrarContorno = True

dosGraficas(    data,
                data2,
                miTitulo,
                miTitulo2,
                miFecha,
                miFecha2,
                miColor,
                puntosMuestreo,
                direccionReloj,
                mostrarContorno)