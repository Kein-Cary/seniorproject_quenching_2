import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
#astropy package for astronomy
from astropy.units import deg,Mpc,m,lyr
from astropy.constants import G,c
from astropy.cosmology import WMAP9
from lenstools.simulations import PotentialPlane,RayTracer
tracer = RayTracer(lens_mesh_size=512)
#Add the lenses to the system
for i in range(11,57):
    tracer.addLens(PotentialPlane.load("D:/Python1/newpython/LensTools/Test/Data/lensing/planes/snap{0}_potentialPlane0_normal0.fits".format(i)))
    tracer.lens[-1].data *= 20
#Rearrange the lenses according to redshift and roll them randomly along the axes
tracer.reorderLenses()
tracer.randomRoll()
WMAP9.comoving_distance(z=2.0)
#These are the initial ray positions as they hit the first lens
sample_ray_bucket = np.array([[0.0,-0.1,0.1,-0.2,0.2],[0.0,0.0,0.0,0.0,0.0]]) * deg
#This method computes the light ray deflections through the lenses and displays them schematically
tracer.displayRays(sample_ray_bucket,z=2.0)
