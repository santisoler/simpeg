from SimPEG import Survey, Utils, Problem, np, sp, mkvc
from scipy.constants import mu_0
import sys
from numpy.lib import recfunctions as recFunc

############
### Data ###
############
class DataMT(Survey.Data):
    '''
    Data class for MTdata 

    :param SimPEG survey object survey: 
    :param v vector with data

    '''
    def __init__(self, survey, v=None):
        # Pass the variables to the "parent" method
        Survey.Data.__init__(self, survey, v)

    def toRecArray(self,returnType='RealImag'):
        '''
        Function that returns a numpy.recarray for a SimpegMT impedance data object.

        :param str returnType: Switches between returning a rec array where the impedance is split to real and imaginary ('RealImag') or is a complex ('Complex')

        '''

        def rec2ndarr(x,dt=float):
            return x.view((dt, len(x.dtype.names)))
        # Define the record fields
        dtRI = [('freq',float),('x',float),('y',float),('z',float),('zxxr',float),('zxxi',float),('zxyr',float),('zxyi',float),('zyxr',float),('zyxi',float),('zyyr',float),('zyyi',float)]
        dtCP = [('freq',float),('x',float),('y',float),('z',float),('zxx',complex),('zxy',complex),('zyx',complex),('zyy',complex)]
        impList = ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi']
        for src in self.survey.srcList:
            # Temp array for all the receivers of the source.
            # Note: needs to be written more generally, using diffterent rxTypes and not all the data at the locaitons
            # Assume the same locs for all RX
            locs = src.rxList[0].locs
            tArrRec = np.concatenate((src.freq*np.ones((locs.shape[0],1)),locs,np.nan*np.ones((locs.shape[0],8))),axis=1).view(dtRI)
            # np.array([(src.freq,rx.locs[0,0],rx.locs[0,1],rx.locs[0,2],np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ) for rx in src.rxList],dtype=dtRI)
            # Get the type and the value for the DataMT object as a list
            typeList = [[rx.rxType,self[src,rx]] for rx in src.rxList]
            # Insert the values to the temp array
            for nr,(key,val) in enumerate(typeList):
                tArrRec[key] = mkvc(val,2)
            # Masked array 
            mArrRec = np.ma.MaskedArray(rec2ndarr(tArrRec),mask=np.isnan(rec2ndarr(tArrRec))).view(dtype=tArrRec.dtype)
            # Unique freq and loc of the masked array
            uniFLmarr = np.unique(mArrRec[['freq','x','y','z']])

            try:
                outTemp = recFunc.stack_arrays((outTemp,mArrRec))
                #outTemp = np.concatenate((outTemp,dataBlock),axis=0)
            except NameError as e:
                outTemp = mArrRec

            if 'RealImag' in returnType:
                outArr = outTemp
            if 'Complex' in returnType:
                # Add the real and imaginary to a complex number

                outArr = np.empty(outTemp.shape,dtype=dtCP)
                for comp in ['freq','x','y','z']:
                    outArr[comp] = outTemp[comp].copy()
                for comp in ['zxx','zxy','zyx','zyy']:
                    outArr[comp] = outTemp[comp+'r'].copy() + 1j*outTemp[comp+'i'].copy()

        # Return 
        return outArr
