# Averaging spatially resolved strain in reconstructed 
# objects of nominally zero strain to determine approximate 
# strain resolution of BCDI measurement.

import numpy as np
import scipy.io as sio

from argparse import Namespace
from scipy.ndimage.morphology import binary_erosion


data = Namespace( 
    **sio.loadmat( 
        '/home/smaddali/ANL/Manuscripts/HEBCDI/data/estimatedStrains_correctScale.mat' 
    ) 
)

##########################################################

rho = data.rho_stdSample
strain = data.strain_stdSample
label=r'Au nanoparticle'
numErosions = 3

#rho = data.rho_111_A
#strain = data.strain_111_A
#label = r'Grain $111$' 
#numErosions = 5

#rho = data.rho_111_B
#strain = data.strain_111_B
#label=r'Grain $\bar{1}\bar{1}\bar{1}$'
#numErosions = 5

histbins = 50

##########################################################

sup = ( np.absolute( rho  ) > 0. ).astype( float )
for n in list( range( numErosions ) ):
    sup = binary_erosion( sup )

straindata = strain[ np.where( sup > 0.5 ) ]


#plt.clf()
plt.hist( 
    straindata, 
    bins=np.linspace( straindata.min(), straindata.max(), histbins ), 
    histtype='step' ,
    linewidth=2, 
    label=label
)



