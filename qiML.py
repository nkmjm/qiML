# for python3
# qiML (quantum-inspired Machine Learning)

import numpy as np


def center_scale(A):

    A_mean = np.mean(A)
    A_std = np.std(A)
    A_nrm = A
    A_nrm -= A_mean
    A_nrm /= A_std

    return A_nrm, A_mean, A_std


#####
##### qiSVD
#####

def vec2strctData(vector):
    
    # Computing prob
    prob = vector**2
    prob = prob/np.sum(prob) 
    
    # Computing depth(d)
    dim = len(vector)
    d = np.log2(dim)
    d = int(np.ceil(d))
    
    # Computing cumProb
    cumProb=np.ones(2**d)
    cumProb[0:len(prob)]=np.cumsum(prob)
    
    output=dict()
    output={'prob': prob,'cumProb':cumProb}

    return output

def mat2strctData(A):
    
    prob = A**2
    prob = np.sum(prob, axis=1)
    prob = prob/np.sum(prob)
    
    # Computing depth(d)
    dim = len(prob)
    d = np.log2(dim)
    d = int(np.ceil(d))
    
    # Computing cumProb
    cumProb=np.ones(2**d)
    cumProb[0:len(prob)] = np.cumsum(prob)
    
    nRow = A.shape[0]
    rowInfo = []
    for rowID in range(0,nRow): 
        tmpInfo = vec2strctData(A[rowID,:])
        rowInfo.append(tmpInfo) 

    output = dict()
    output = {'rawMat':A, 'prob':prob, 'cumProb':cumProb, 'infoOnEachRow':rowInfo}

    return output

def samplingFromTree(cumProb,z):
    
    # Computing d (depth)
    dim = len(cumProb)
    d = np.log2(dim)
    d = int(np.ceil(d))
    
    # Explore end node corresponding to current_z
    # ---------------------------------------------------
    # I (threshold ID)
    # temporal_dimID (obtained end-node ID)
    # dimID (end-node IDs corresponding to z)
    # ---------------------------------------------------
    
    dimID=list()
    for current_z in z:
        I = (2** (d-1)) - 1
        temporal_dimID = 0
        for index_d in range(d):
            threshold=cumProb[I]
            if current_z<threshold:
                I = I - 2** (d-(index_d+2))
            else:
                temporal_dimID = temporal_dimID + 2** (d-index_d-1)
                I = I + 2** (d-(index_d+2))
        dimID.append(temporal_dimID)
        
    return dimID

def delete_small_singlar_values(s):

    minS = (10** (-10))
    new_k_IDs = s[s > minS] # using only components of [S > minS] 
    new_k = len(new_k_IDs)
    s = s[0:new_k]

    return s, new_k

def qiSVD(A, k, p, compute_uv=True, returnParams=False, normalizeData=False):

    '''
    (def qiSVD)

    quantum-inspired Singular Value Decomposition

    Structure of input and return data is referred to 
    scipy.linalg.decomp_svd.py (https://github.com/scipy/scipy/blob/v1.3.0/scipy/linalg/decomp_svd.py#L16-L139)

    Parameters
    ----------
    A : (M, N) array_like. Set Matrix to decompose.
    k : int. The number of components.
    p : int. The number of tree-sampling.
    compute_uv : bool (default is True).
                 If True, U and Vh are computed in addition to s.
    returnParam: bool (default is False).
                 If True, params (dimID and U) are returned
    normalizeData : bool (default is False).
                    If True, input A is normalized.

    Returns
    -------
    U : (M, K) ndarray. Matrix of the left singular vectors.
    s : (1, K) ndarray. The singular values that are sorted in non-increasing order.
    Vh: (K, N) ndarray. Matrix of the right singular vectors.
    '''

    # Nomalizing A if normalizeData is True
    if normalizeData is True:
        _, A_mean, _ = center_scale(A)
        A -= A_mean

    # Setting structure data of A
    data = mat2strctData(A)

    # Sampling for rowIDs
    treeData = data['cumProb']
    z = np.random.rand(p)
    rowIDs = samplingFromTree(treeData, z)

    # Sampling for columnIDs
    rowInfo = data['infoOnEachRow']
    columnIDs = []
    for pi in range(0, p):
        tmpID = np.random.randint(0, p)
        tmpRowID = rowIDs[tmpID]
        tmpRowData = rowInfo[tmpRowID]#this may be slow. can we combined the below line?
        tmpTreeData = tmpRowData['cumProb']
        tmpZ = np.random.rand(1)
        tmpColumnID = samplingFromTree(tmpTreeData, tmpZ)
        columnIDs.append(tmpColumnID[0])
   
    # Setting p x p matrix
    ppMt = []
    rowProb = data['prob']#This may be slow.
    probMt = []
    for ri in rowIDs:
        tmpRowX = data['rawMat'][ri, columnIDs]
        tmpRowProb = rowInfo[ri]['prob'][columnIDs]
        # Set matrices
        ppMt.append(tmpRowX)
        probMt.append(tmpRowProb)
    
    # list to np.array
    ppMt = np.array(ppMt)
    probMt = np.array(probMt)
    
    # Normalize ppMt
    sqrtD = np.sqrt(rowProb[rowIDs])*np.sqrt(p)
    sqrtF  = np.sqrt(np.mean(probMt, axis=0))*np.sqrt(p)#In the code before 20190509, sqrt(p) is not multiplied.
    ppMt = ppMt.transpose()/sqrtD
    ppMt = ppMt.transpose()/sqrtF
  
    # Computing conventional SVD
    print('SVD ppMt')
    u, s, _ = np.linalg.svd(ppMt) 
    # Error catch : k < p
    if ppMt.shape[1] < k:
        print('k should be less than p.')
    
    # Obtaining the first k components 
    u = u[:,0:k]
    s = s[0:k]
    
    # Deleting the singular vectors corresponding to small singular values.
    s, new_k = delete_small_singlar_values(s)
    u = u[:, 0:new_k]

    # Normalizing u
    norm_u = u/s
    norm_u = norm_u.transpose()/np.sqrt(p* rowProb[rowIDs])
    norm_u = norm_u.transpose()
    #scale = np.sqrt(p* rowProb[rowIDs])

    # Orthonormalization: using Schmidt's 
    print('Orthomalization')
    A =  data['rawMat'][rowIDs, :]#This may be slow.
    AA=np.dot(A, A.transpose())

    U = np.zeros_like(norm_u)
    u0 = norm_u[:,0]
    
    errThres = 10**(-5)
    
    U[:,0] = u0/np.sqrt(np.dot(np.dot(u0.T, AA), u0))
    for index_k in range(1, new_k):
        coef = np.dot(np.dot(U[:,range(index_k)].T, AA), norm_u[:,index_k])
        U[:, index_k]=norm_u[:, index_k] - np.dot(U[:, range(index_k)], coef)
        U[:, index_k]=U[:, index_k]/np.sqrt(np.dot(np.dot(U[:,index_k].T, AA), U[:,index_k]))
        vvMat = np.dot(np.dot(U[:, range(index_k+1)].T, AA), U[:, range(index_k+1)])
        eyeMat = np.eye(index_k+1)
        err = np.sum((vvMat - eyeMat)**2)
        # Error catch : error < errThres
        if err > errThres:
            U[:, index_k] = np.nan
        
    # Removing nanInds
    Uinds = list(range(1, new_k))
    us = np.sum(U, axis=0)
    rmUinds = np.where(np.isnan(us), 1, 0) 
    U = U[:, rmUinds==0]
    
    # Computing Vh
    Vh = np.dot(U.transpose(), A)

    if returnParams:
        svdParams = dict()
        svdParams = {'dimID':rowIDs, 'U':U}
        return svdParams

    return U, s, Vh


#####
##### qiCCA
#####

def calcResultCCAfromV(X, Y, Vx, Vy, k, compute_wv=True):
    
    C = np.dot(Vx, Vy.transpose())
    
    U, _, V = np.linalg.svd(C) # U: nCSample(=k) x nCK(=k), V: nCK(=k) x nCDim(=k)
    U = U[:, 0:k]
    V = V[0:k, :]
    
    if compute_wv is True:
        
        # Estimating U, S
        UxSx = np.dot(X[:, :], Vx.transpose()) # nDim x k_x
        Sx   = np.sqrt(np.sum(UxSx**2, axis=0)) # k_x
        Ux   = UxSx/Sx.transpose() # nDim x k_x
        UySy = np.dot(Y[:, :], Vy.transpose())
        Sy   =  np.sqrt(np.sum(UySy**2, axis=0))
        Uy   = UySy/Sy.transpose()

        A = np.dot(Ux/Sx, U) #v* s-1*U
        B = np.dot(Uy/Sy, V.transpose())
    
        # Computing canonCompX,canonCompY using weight matrices (A,B)
        canonCompX_usingA = np.dot(X[:, :].transpose(), A) # nSample x k
        canonCompY_usingB = np.dot(Y[:, :].transpose(), B) # nSample x k
        ccMat_usingAB = np.corrcoef(canonCompX_usingA.transpose(), canonCompY_usingB.transpose()) # k x k
        
    # Computing canonCompX,canonCompY using only Vx, Vy
    canonCompX = np.dot(Vx.transpose(), U) # nSample x k
    canonCompY = np.dot(Vy.transpose(), V.transpose()) # nSample x k
    
    if compute_wv is True:
        out = {'canonCompX':canonCompX, 'canonCompY':canonCompY, 'A':A, 'B':B, 'U':U, 'V':V}
    else:
        out = {'canonCompX':canonCompX, 'canonCompY':canonCompY}
    
    return out

def corrcoef(x_scores, y_scores):

    '''
    Parameters
    ----------
    x_scores : (N, k) ndarray. Canonical component of training vectors.
    y_scores : (N, k) ndarray. Canonical component of target vectors.

    Returns
    ----------
    rs : (k, ) ndarray. Correlation coefficient between x_scores and y_scores for each component.
    '''

    rMat = np.corrcoef(x_scores.T, y_scores.T)
    k = int(rMat.shape[0]/ 2)
    rs = np.diag(rMat[0:k, k:2*k])

    return rs

class qiCCA():

    def __init__(self, k_x=1000, k_y=1000, k=100, p=120, compute_wv=True, normalizeData=False):

        self.k_x = k_x
        self.k_y = k_y
        self.k = k
        self.p = p
        self.compute_wv = compute_wv
        self.normalizeData = normalizeData

    def fit(self, X, Y):

        '''
        Parameters
        ----------
        X : (N, D1) ndarray. Training vectors. D1 denotes the number of features.
        Y : (N, D2) ndarray. Target vectors. D2 denotes the number of targets.
        compute_wv : bool (default is False).
                     If True, weight vectors for X and Y are returned.
        normalizeData : bool (default is False).
                        If True, input X and Y are normalized.

        Returns
        ----------
        x_scores : (N, k) ndarray. Canonical components for X.
        y_scores : (N, k) ndarray. Canonical components for Y.
        x_weights: (N, k_x) ndarray. Weight vectors for X to project into the shared space.
        y_weights: (N, k_y) ndarray. Weight vectors for Y to project into the shared space.
        self.dimID_x  : (p, ) list. Sampled indices for X.
        self.dimID_y  : (p, ) list. Sampled indices for Y.
        self.corrcoefs: (k, ) ndarray. Correlation coefficients between x_scores and y_scores.
        '''

        k_x = self.k_x
        k_y = self.k_y
        k = self.k
        p = self.p

        # Transpose X, Y
        X = X.T
        Y = Y.T

        # Nomalizing X, Y
        if self.normalizeData is True:
            X, self.x_mean_, self.x_std_ = center_scale(X)
            Y, self.y_mean_, self.y_std_ = center_scale(Y)

        # Setting structure data of X, Y
        data_x = mat2strctData(X)
        data_y = mat2strctData(Y)
        
        # Computing qiSVD
        print('Performing qiSVD on X')
        svdParams_x = qiSVD(X, k_x, p, returnParams=True)
        print('Performing qiSVD on Y')
        svdParams_y = qiSVD(Y, k_y, p, returnParams=True) 
        
        # Setting data
        dimID_x = svdParams_x['dimID']
        dimID_y = svdParams_y['dimID']  
        discriptionUx = svdParams_x['U']
        discriptionUy = svdParams_y['U']
        X = data_x['rawMat']
        Y = data_y['rawMat']
        newK = discriptionUx.shape[1]
        
        # Reset nSample of X and Y
        nOrgSample = data_x['rawMat'].shape[1]
        
        # Computing Vx, Vy
        Vx = np.dot(discriptionUx.transpose(), X[dimID_x, :]) # k_x x nSample
        Vy = np.dot(discriptionUy.transpose(), Y[dimID_y, :]) # k_y x nSample
        
        # Computing canonCompX,canonCompY using Vx, Vy
        out = calcResultCCAfromV(X[dimID_x, :], Y[dimID_y, :], Vx, Vy, k, self.compute_wv)

        if self.compute_wv is True:
            from scipy.linalg import pinv2
            self.x_weights = np.dot(discriptionUx, out['U'])
            self.y_weights = np.dot(discriptionUy, out['V'].transpose())

        # Results matrices
        self.x_scores = out['canonCompX']
        self.y_scores = out['canonCompY']

        # Setting dimID_x, _y
        self.dimID_x = dimID_x
        self.dimID_y = dimID_y

        # Computing corrcoef
        self.corrcoefs = corrcoef(self.x_scores, self.y_scores)
            
        return self

    def transform(self, X, Y=None, copy=True):

        '''
        Parameters
        ----------
        X : (N, D1) ndarray. Test vectors. D1 denotes the number of features.
        Y : (N, D2) ndarray. Target vectors. D2 denotes the number of targets.

        Returns
        ----------
        x_scores : (N, k) ndarray. Canonical components for X.
        y_scores : (N, k) ndarray. Canonical components for Y.
        '''

        # Transpose X, Y
        X = X.T
        Y = Y.T

        # Normalizing X if self.normalizeData is True
        if self.normalizeData is True:
            X -= self.x_mean_
            X /= self.x_std_

        # Computing x_scores (and y_scores)
        x_scores = np.dot(X[self.dimID_x, :].T, self.x_weights)
        if Y is not None:
            # Normalizing Y if self.normalizeData is True
            if self.normalizeData is True:
                Y -= self.y_mean_
                Y /= self.y_std_
            y_scores = np.dot(Y[self.dimID_y, :].T, self.y_weights)
            return x_scores, y_scores

        return x_scores, y_scores

    def fit_transform(self, X, Y):

        '''
        Parameters
        ----------
        X : (N, D1) ndarray. Training vectors. D1 denotes the number of features.
        Y : (N, D2) ndarray. Target vectors. D2 denotes the number of targets.

        Returns
        ----------
        x_scores : (N, k) ndarray. Canonical components for X.
        y_scores : (N, k) ndarray. Canonical components for Y.
        '''

        return self.fit(X, Y).transform(X, Y)

