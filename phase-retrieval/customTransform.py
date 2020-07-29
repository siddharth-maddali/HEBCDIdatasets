import scipy.fftpack as spfft

def customDCT2( img ):
    return spfft.dct( 
            spfft.dct( img.T, norm='ortho', axis=0 ).T, 
            norm='ortho', axis=0 
    )

def inverseDCT2( img ):
    return spfft.idct( 
            spfft.idct( img.T, norm='ortho', axis=0 ).T, 
            norm='ortho', axis=0
    )
