dat_A = Namespace( **sio.loadmat( '/home/smaddali/ANL/Manuscripts/manuscripts-HEBCDI/data/111_A_solution.mat' ) )
dat_B = Namespace( **sio.loadmat( '/home/smaddali/ANL/Manuscripts/manuscripts-HEBCDI/data/111_B_solution.mat' ) )
dat_T = Namespace( **sio.loadmat( '/home/smaddali/ANL/Manuscripts/manuscripts-HEBCDI/data/stdSample_solution.mat' ) )
dat_S = Namespace( **sio.loadmat( '/home/smaddali/ANL/Manuscripts/manuscripts-HEBCDI/data/estimatedStrains_smooth.mat' ) )

################################################################################################################

phaseA = np.angle( dat_A.rho.ravel() )
hereA = np.where( phaseA < 0. )
phaseA[ hereA ] = phaseA[ hereA ] + 2.*np.pi
phaseA = phaseA - np.pi

phaseB = -np.angle( dat_A.rho.ravel() ) # -ve sign for twin plot

# next 4 lines: shifting away of garbage phases
shiftThese =np.where( phaseB==0. )
phaseB[ shiftThese ] = phaseB[ shiftThese ] + 2.*np.pi 
shiftThese =np.where( phaseB==-np.pi )
phaseB[ shiftThese ] = phaseB[ shiftThese ] + 3.*np.pi 

hereB = np.where( phaseB < 0. )
phaseB[ hereB ] = phaseB[ hereB ] + 2.*np.pi
phaseB = phaseB - np.pi

B = np.linspace( -2.5, 2.5, 80 )
Bstrain = np.linspace( -0.007, 0.007, 80 )

# for std sample
binsStd = np.linspace( dat_S.strain_stdSample.min(), dat_S.strain_stdSample.max(), 100 )

################################################################################################################

plt.figure( 1 )
plt.clf()
plt.hist( phaseA, bins=B, weights=np.absolute( dat_A.rho.ravel() ), histtype='step', label=r'$111$', linewidth=2 )
plt.hist( phaseB, bins=B, weights=np.absolute( dat_B.rho.ravel() ), histtype='step', label=r'$\bar{1}\bar{1}\bar{1}$', linewidth=2 )
plt.rc( 'font', **{ 'size':18, 'weight':'bold' } )
plt.xlabel( r'$\phi$ (radians)', fontsize=18, fontweight='bold' )
plt.ylabel( r'Weighted voxel count (arb. units)', fontsize=18, fontweight='bold' )
plt.grid()
plt.legend( loc='best' )
plt.rc( 'legend', **{ 'fontsize':18 } )
plt.show()
#plt.savefig( '/home/smaddali/ANL/Manuscripts/manuscripts-HEBCDI/indivFigures/phaseHistograms.pdf' )

plt.figure( 2 )
plt.clf()
plt.hist( 
    dat_S.strain_111_A.ravel(), 
    bins=Bstrain, 
#    weights=np.absolute( dat_S.strain_111_A.ravel() ), 
    histtype='step', 
    label=r'$111$', 
    linewidth=2 
)
plt.hist( 
    dat_S.strain_111_B.ravel(), 
    bins=Bstrain, 
#    weights=np.absolute( dat_S.strain_111_B.ravel() ), 
    histtype='step', 
    label=r'$\bar{1}\bar{1}\bar{1}$', 
    linewidth=2
)
plt.yscale( 'log' )
plt.rc( 'font', **{ 'size':18, 'weight':'bold' } )
plt.xlabel( r'$\partial u_{hkl} / \partial x_{hkl}$', fontsize=18, fontweight='bold' )
plt.ylabel( r'Weighted voxel count (arb. units)', fontsize=18, fontweight='bold' )

plt.grid()
plt.legend( loc='best' )
#plt.savefig( '/home/smaddali/ANL/Manuscripts/manuscripts-HEBCDI/indivFigures/strainHistograms.pdf' )


plt.figure( 3 ) # for std sample
plt.hist( 
    dat_S.strain_stdSample.ravel(), 
    bins=binsStd, 
    weights=np.absolute( dat_S.rho_stdSample.ravel() ), 
    histtype='step', 
    label=r'Au nanoparticle', 
    linewidth=2
)
plt.xlabel( r'$\partial u_{111} / \partial x_{111}$' )
plt.ylabel( r'Weighted voxel count (arb. units)', fontsize=18, fontweight='bold' )
plt.grid()
plt.legend( loc='best' )
