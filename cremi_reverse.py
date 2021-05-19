######## IMPORTS & Helpful Functions ########
import numpy as np
import imageio
import math
import gc, argparse, sys, os, errno
import h5py
import warnings
warnings.filterwarnings('ignore')
import scipy
from scipy.ndimage.measurements import label
from skimage.measure import regionprops
import cv2
from shutil import copyfile

def readh5(filename, datasetname=None):
    import h5py
    fid = h5py.File(filename,'r')

    if datasetname is None:
        if sys.version[0]=='2': # py2
            datasetname = fid.keys()
        else: # py3
            datasetname = list(fid)
    if len(datasetname) == 1:
        datasetname = datasetname[0]
    if isinstance(datasetname, (list,)):
        out=[None]*len(datasetname)
        for di,d in enumerate(datasetname):
            out[di] = np.array(fid[d])
        return out
    else:
        return np.array(fid[datasetname])

def writeh5(filename, dtarray, datasetname='main'):
    import h5py
    fid=h5py.File(filename,'w')
    if isinstance(datasetname, (list,)):
        for i,dd in enumerate(datasetname):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()

def get_roi(image,anotherimage):
    img = np.copy(image)
    img_ = np.copy(image)
    labels = label(image)
    img[np.where(labels[0]==1)] =0
    img_[np.where(labels[0]==2)] =0
    anoimg = np.copy(anotherimage)
    anoimg_ = np.copy(anotherimage)
    anoimg[np.where(labels[0]==1)] =0
    anoimg_[np.where(labels[0]==2)] =0
    return anoimg,anoimg_

def convert_to_uint8(filepath):
    with h5py.File(filepath) as f:
        data = f['main'][:].astype('uint8')
    os.system('rm '+filepath)
    with h5py.File(filepath) as f:
        f.create_dataset('main',data=data)

######## Input Properties ########
crack_path = 'crack/'# output path of the images with crack (no need to change)
trial = 'result04/'  # working directory
num_channels = 1     # Default channels

# Sanity Check
if not os.path.exists('prediction/'+trial):
    os.makedirs('prediction/'+trial)
if not os.path.exists('reverse/'+trial):
    os.makedirs('reverse/'+trial)
    
#Paths of Vol's to be reversed
pred_file_path = 'prediction/'+trial
pred_A = 'im_A+_v2_200_nocrack_pred.h5'
pred_B = 'im_B+_v2_200_pred.h5'
pred_C = 'im_C+_v2_200_pred.h5'


######## Vol A resolve ########
def deal_with_crack():
    """Uses SIFT and Affine Transforms to fit 15th and 48th prediction"""
    ind = [15,48]
    vid = 0
    for i in ind:
        for j in range(1, 3):
            im_path = 'crack/tmp/im0' + str(i) + '_prediction' + str(j) + '.png'
            im = imageio.imread(im_path)
            sz = im.shape
            #print("Size of image: ", sz)
            vol_path = 'crack/align_' + str(vid) + '_' + str(i) + '_' + str(j) + '.hdf'
            tmp = readh5(vol_path)
            tmpp = tmp[1, :, :]
            tmpp_inv = np.linalg.inv(tmpp)
            trans_matrix = tmpp_inv[:2, :]
            if j == 1:
                B = cv2.warpAffine(im, trans_matrix, dsize = (1912, 1741), flags = cv2.INTER_LINEAR)
            else:
                B = B + cv2.warpAffine(im, trans_matrix, dsize = (1912, 1741), flags = cv2.INTER_LINEAR)
        save_path = 'crack/tmp/im_0' + str(i)+ '_reverse0.png'
        imageio.imwrite(save_path, B)
    
    
def crack_fix(path_A):
    vol = readh5(path_A)
    # Extract 14th and 47th slice
    prediction_14 = vol[0, 14, :, :]
    prediction_47 = vol[0, 47, :, :]
    # Extract ROI and save as PNG
    ## Sanity Check
    if os.path.exists(crack_path+'tmp'):
        os.system('rm -rf '+crack_path+'tmp')
        os.mkdir(crack_path+'tmp')
    else:
        os.mkdir(crack_path+'tmp')
    imageio.imwrite(crack_path+'tmp/im015_prediction1.png',get_roi(imageio.imread(crack_path+'im015_warp0.png'),prediction_14.T)[1])
    imageio.imwrite(crack_path+'tmp/im015_prediction2.png',get_roi(imageio.imread(crack_path+'im015_warp0.png'),prediction_14.T)[0])
    imageio.imwrite(crack_path+'tmp/im048_prediction1.png',get_roi(imageio.imread(crack_path+'im048_warp0.png'),prediction_47.T)[1])
    imageio.imwrite(crack_path+'tmp/im048_prediction2.png',get_roi(imageio.imread(crack_path+'im048_warp0.png'),prediction_47.T)[0])
    
    deal_with_crack()
    
    with h5py.File(pred_file_path+pred_A) as f:
        resultsA_reverse1 = f['main'][:]
        resultsA_reverse1[0, 14, :, :] = imageio.imread(crack_path+'tmp/im_015_reverse0.png').T
        resultsA_reverse1[0, 47, :, :] = imageio.imread(crack_path+'tmp/im_048_reverse0.png').T
        writeh5(pred_file_path+'im_A+_v2_200_pred.h5', resultsA_reverse1)
        #with h5py.File(pred_file_path+'im_A+_v2_200_pred.h5') as f:
            #f.create_dataset('main',data=resultsA_reverse1)

def bad_slices_fix(vol_idx, channel_num):
    """Fixes wrong 15th and 79th slice in Vol A+ and Vol B+"""
    for i in vol_idx:
        if i == 0:
            vol_name = "A+"
        elif i == 1:
            vol_name = "B+"
        elif i == 2:
            continue
        vol_path = 'reverse/result04/results_new_' + vol_name + '_v2_200_' + str(channel_num) + '.h5'
        print("Fixing bad slices in: ", vol_path)
        vol = readh5(vol_path, 'main')
        _14_slice = vol[14, :, :].copy()
        old_15_slice = vol[15, :, :].copy()
        vol[15, :, :] = _14_slice
        new_15_slice = vol[15, :, :]

        _78_slice = vol[78, :, :].copy()
        old_79_slice = vol[79, :, :].copy()
        vol[79, :, :] = _78_slice
        new_79_slice = vol[79, :, :]

        if np.array_equal(new_15_slice, old_15_slice):
            print("14->15 done incorrectly")
        else:
            print("14 -> 15 Swap done right!")

        if np.array_equal(new_79_slice, old_79_slice):
            print("78->79 done incorrectly")
        else:
            print("78 -> 79 Swap done right!")
        writeh5(vol_path, vol)
   
        
        
def threshold_vol(vol_idx, channel_num, thres = 0.5):
    for i in vol_idx:
        if i == 0:
            vol_name = "A+"
        elif i == 1:
            vol_name = "B+"
        elif i == 2:
            vol_name = "C+"
        vol_path = 'reverse/result04/results_new_' + vol_name + '_v2_200_' + str(channel_num) + '.h5'
        vol = readh5(vol_path)
        threshold_limit = thres*256
        vol_mod = (vol>threshold_limit).astype(np.uint8)
        writeh5(vol_path, vol_mod)
        
def reverse_all(vol_idx, channel_num):
    # volume names
    nn=['A','B','C','A+','B+','C+']
    # bad slices
    bb=[[143],[1,29,30,58,59,91],[28,88,100],[65,93,94,122,123,125],[1,29,30,58,59,91],[28,88,100]]  
    # to be replaced
    gg=[[142],[0,28,31,57,60,90],[27,87,99],[64,92,95,121,124,126],[0,28,31,57,60,90],[27,87,99]]
    # newly-aligned image size
    sz=[[1727,1842],[2069,1748],[1986,2036],[1741,1912],[2898,1937],[1914,1983]]

    # CREMI: 125,1250,1250
    # _v2_200: 200 margin from manual label
    suf='v2_200'
    vol_idx_new = [x+3 for x in vol_idx]
    #print("Vol Idx New: ", vol_idx_new)
    for nid in vol_idx_new: 
        trial = 'result04/'
        vol = nn[nid]
        sn='05'
        if len(vol)==2:
            sn = "06"
        pw=0
        ph=0 
        if vol == 'B+':
            ph=700
        syn_warp = readh5('prediction/' + trial + 'im_'+ vol + '_v2_200_pred.h5','/main')
        syn_warp = syn_warp[channel_num, :, :, :]

        sz_r = syn_warp.shape #Shape of the vol
        sz_r = tuple((sz_r))
        new_size = sz[nid]
        new_size = np.array(new_size)
        n_size = np.flipud(new_size) 

        modified_vol_shape = n_size - 400
        #print('modified vol shape: ', modified_vol_shape)
        temp = np.insert(modified_vol_shape, 0, 125, axis=0)

        sz_bad = sz_r - temp
        sz_bad = sz_bad//2
        #print('sz_bad: ', sz_bad)
        #print('syn_warp old shape: ', syn_warp.shape)
        syn_warp = syn_warp[sz_bad[0]: -sz_bad[0], sz_bad[1]: -sz_bad[1], sz_bad[2]: -sz_bad[2]]
        #print('syn_warp new shape: ', syn_warp.shape)

        pp=np.loadtxt('align/trans_' + vol + '_v2.txt', delimiter = ',')
        pp = np.cumsum(pp, 0)

        _77thElement = pp[76, :]
        pp = pp - _77thElement
        pp = -pp #Since it's -bsxfun()

        max1, max2 = pp.max(axis = 0)
        min1, min2 = pp.min(axis = 0)

        max1 = math.ceil(max1)
        max2 = math.ceil(max2)
        min1 = math.ceil(-min1)
        min2 = math.ceil(-min2)

        ww = np.array([max1, max2, min1, min2])
        result_o = np.zeros((125,1250,1250),dtype = np.int16)
        for i in range(0, 125):
            pd = np.round(pp[14+i])
            tmp = np.zeros((3072, 3072))
            tmp2 = np.pad(tmp, pad_width = ((0, 0), (ph, ph)), mode = 'symmetric')

            x_start = int(911 + pd[0] - ww[0] + ph)
            x_end = int(911 + pd[0] + 1250 + ww[2] + ph)
            y_start = int(911 + pd[1] - ww[1] + pw) 
            y_end = int(911 + pd[1] + 1250 + ww[3] + pw)

            tmp2[y_start:y_end, x_start:x_end] = syn_warp[i, :, :]
            result_o[i, :, : ] = tmp2[pw + 911 : pw + 911 + 1250, ph + 911 : ph + 911 + 1250]

        writeh5('reverse/' + trial + 'results_new_' + vol + '_' + suf + '_' + str(channel_num) + '.h5',result_o, 'main')

def type_conversion(vol_idx, channel_num):
    for i in vol_idx:
        if i == 0:
            vol_name = "A+"
        elif i == 1:
            vol_name = "B+"
        elif i == 2:
            vol_name = "C+"
        vol_path = 'reverse/result04/results_new_' + vol_name + '_v2_200_' + str(channel_num) + '.h5'
        convert_to_uint8(vol_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description='CREMI Reverse Script')

    parser.add_argument('-a', '--pathA', help='Filepath for VolA', default='.', type=str)
    parser.add_argument('-b', '--pathB', help='Filepath for VolB', default='.', type=str)
    parser.add_argument('-c', '--pathC', help='Filepath for VolC', default='.', type=str)
    parser.add_argument('-thres', '--threshold', help='Threshold for volume generation', default=0.5, type=float)
    args = parser.parse_args()
    return args

def file_copy(filePath, volIdx):
    if volIdx == 0: #volA
        copyfile(filePath, pred_file_path + 'im_A+_v2_200_nocrack_pred.h5')
    if volIdx == 1: #volB
        copyfile(filePath, pred_file_path + 'im_B+_v2_200_pred.h5')
    if volIdx == 2:
        copyfile(filePath, pred_file_path + 'im_C+_v2_200_pred.h5')

def get_num_channels(file_path):
    vol = readh5(file_path)
    num_channels = vol.shape[0]
    return num_channels
    
if __name__ == "__main__":
    #0/1/2 corresponding to vols
    args = parse_arguments()
    vol_idx = [] 
    if args.pathA != ".":
        vol_idx.append(0)
        file_copy(args.pathA, 0)
    if args.pathB != ".":
        vol_idx.append(1)
        file_copy(args.pathB, 1)
    if args.pathC != ".":
        vol_idx.append(2)
        file_copy(args.pathC, 2)
    
    # Fix cracks in first channel of volA. This Assumes First channel contains probability map
    if 0 in vol_idx:
        print("Dealing with Cracks in Vol A")
        crack_fix(pred_file_path+pred_A)
    # Calculate number of channels
    num_channels = get_num_channels(args.pathA)
    print("Number of channels: ", num_channels)
    for channel in range(num_channels):
        print('\n')
        print(f"Dealing with channel: {channel}")
        print("#"*40)
        print("Reversing Volumes Now....")
        reverse_all(vol_idx, channel)
        print("Converting to np.uint8....")
        type_conversion(vol_idx, channel)
        print("Bad slices are being fixed...")
        bad_slices_fix(vol_idx, channel)
        print("Thresholding for final submission...")
        threshold_vol(vol_idx, channel, args.threshold)
    print("Done!")
    print("Reversed volumes available at reverse/result04/")