import matplotlib 
#matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
#import matplotlib.image as mpimg
import matplotlib.cm as cm
from   matplotlib.colorbar import ColorbarBase
from   matplotlib.colors import Normalize
from   matplotlib.pyplot import figure
from   scipy.interpolate import griddata
import pandas as pd
import numpy as np 
import math
import os

import numpy as np
import matplotlib.pyplot as plt


##################################################################################
def extraeColumnas(datos):
  
  azimuth   = np.array(datos['azimuth'])
  elevation = np.array(datos['elevation'])
  magnitude = np.array(datos['magnitude'])
 
  return((azimuth,elevation,magnitude))
  
################################################################################
def fixMagnitude(magnitude):
  
  inx0    = np.arange(0,6)
  
  inx360  = np.arange(len(magnitude) - 6,len(magnitude))
  
  magnitude0    = magnitude[inx0]
  
  magnitude360  = magnitude[inx360]
  
  magnitudePromedio = (magnitude0 + magnitude360)/2
  
  magnitude[inx0]   = magnitudePromedio 
  
  magnitude[inx360] = magnitudePromedio    

  return(magnitude)
  
################################################################################
# funcion para interpolar los datos 

def interpolaDatos(THETA,R,magnitude):
  
  THETAi  = np.linspace(min(THETA),max(THETA),num=1000.0)
  
  Ri      = np.linspace(min(R),max(R),num=360.0)
  
  theta,r = np.meshgrid(THETAi,Ri,indexing='xy') 
  
  z = griddata((THETA,R),magnitude,(theta,r),method="linear")
  
  z = np.round(z,2)
  
  return((THETAi,Ri,z))
  
################################################################################  
# funcion para seleccinar tabla de colores

def seleccionaColor(miColor):
  if (miColor == "azul"):
    micolor = cm.YlGnBu
  elif (miColor == "anaranjado"):
    micolor = cm.autumn
  elif (miColor == "gris"):
    micolor = cm.Greys 
  return(micolor)
  
################################################################################ 
# para especificar el tamano del plot 
  
fig = figure(figsize=(11.0,5.9))

################################################################################
def polarPlot(df,
              miTitulo,
              miFecha,
              posicion,
              miColor="azul",
              puntosMuestreo = False,
              direccionReloj = False,
              mostrarContorno = False,
              Repetir = False):

                  


  # extraer columnas del dataframe
  azimuth,elevation,magnitude = extraeColumnas(df)
   
  # para seleccinar tabla de colores

  micolor = seleccionaColor(miColor)
    
  # para promediar la medidas a azimuto 0 y 360
    
  magnitude = fixMagnitude(magnitude)
    
  # conversion de angulos a radianes 
      
  theta = np.radians(azimuth)
  phi   = np.radians(elevation)
  
  # definicion de eje para barra de colores 
  
  cax = fig.add_axes([0.25, 0.05, 0.5, 0.05])
  
  # definicion de eje para grafica polar 
  
  ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], 
                  frameon=True,
                  polar=True,axisbg='#d5de9c',aspect='equal')               
    
  thetai,phii,z = interpolaDatos(theta,phi,magnitude)
         
  ax = plt.subplot(posicion, projection='polar')
    
  vmin = 18.0
  vmax = 22.0
    
  ax.contourf(thetai,phii,z,10,linewidth=2,cmap=micolor,vmin=vmin,vmax=vmax)
    
  if direccionReloj:
    ang = 0.0
  else:
    ang = math.pi/2
    
  labels = np.unique(theta)
  
  ax.set_xticks(labels[:-1])
  ax.set_yticks(())
  
  rCard  = 1.4
    
    
  ax.text(ang,1.6,miTitulo,fontsize=16,horizontalalignment='center')
  ax.text(ang,1.4,miFecha,fontsize=16,horizontalalignment='center')
    
  if mostrarContorno:
    ax.contour(thetai,phii,z,10,colors="k")
  
  # mostrar puntos de muestreo 
  
  if puntosMuestreo:
    ax.scatter(theta,phi)  #marker='^' o 's' para cambiar figura de marca
    
  # especificar radio del mapa polar 
  
  ax.set_rmax(1.2)  #maximo
  ax.set_rmin(0)    #minimo
  
  if direccionReloj:  
    ax.set_theta_direction(-1)
    ax.set_theta_offset(math.pi/2.0)
  
  
  
  # para mostrar las unidades al lado de la barra de colores 
  if Repetir:
    if direccionReloj:
      ax.text(np.radians(151),1.7,"$mag/arcsec^2$",
        weight="bold",
        fontsize=16)
    else:
      ax.text(np.radians(299),1.66,"$mag/arcsec^2$",
              weight="bold",
              fontsize=16)
    
  ax.grid(True)
    
  ax.set_xticklabels(['N', '$30^o$', '$60^o$', 'E', 
   '$120^o$', '$150^o$','S', '$210^o$', '$240^o$','W', '$300^o$','$330^o$'],
   ha='center') 
    
  ColorbarBase(cax,orientation='horizontal',cmap=micolor,
               spacing='proportional',
               norm=Normalize(vmin=vmin, vmax=vmax))
               
  if Repetir:
  
    # segunda grafica

    # leer archivo de ejemplo 2
    data = np.loadtxt("ejemplo2.dat")

    # convertir los datos a un DataFrame
    datos = pd.DataFrame(data,columns=['azimuth','elevation','magnitude'])

    #outfile   = "ejemplo2.png"
    miTitulo  = "Ejemplo2.png"
    miFecha   = "00/00/0000"
    miColor   = "azul"
    posicion = 122
    puntosMuestreo  = puntosMuestreo 
    direccionReloj  = direccionReloj 
    mostrarContorno = mostrarContorno 
    Repetir = False

    # para llamar la funcion polarPlot utilizando las opciones por defecto 

    polarPlot(datos,miTitulo, miFecha, posicion, miColor, 
              puntosMuestreo,direccionReloj,mostrarContorno, Repetir)
               
      
  return (plt.show())
  
################################################################################  
## primera grafica

# leer archivo de ejemplo 1
data = np.loadtxt("ejemplo.dat")

# convertir los datos a un DataFrame
datos = pd.DataFrame(data,columns=['azimuth','elevation','magnitude'])

outfile   = "ejemplo.png"
miTitulo  = "Ejemplo.png"
miFecha   = "00/00/0000"
miColor   = "azul"
posicion = 121
puntosMuestreo  = True 
direccionReloj  = True 
mostrarContorno = True
Repetir = True 

# para llamar la funcion polarPlot utilizando las opciones por defecto 

polarPlot(datos,miTitulo, miFecha, posicion, miColor, 
          puntosMuestreo, direccionReloj,mostrarContorno, Repetir)






