from multiprocessing import Pool
from tqdm import tqdm
#
import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt
#
from random import choice, sample, shuffle, uniform, seed
from math import exp, expm1, log1p, log10, log2, sqrt, ceil, floor, isfinite, isnan
from itertools import combinations, compress
#import for image processing
from scipy.stats import kurtosis, skew
from scipy.ndimage import laplace, sobel




def img_to_stats(paths):
    img_id, img = paths[0], paths[1]

    # ignored error
    np.seterr(divide='ignore', invalid='ignore')

    bins = 20
    scl_min, scl_max = -50, 50
    opt_poly = True
    # opt_poly = False

    try:
        st = []
        st_interv = []
        hist_interv = []
        for i in range(img.shape[2]):
            img_sub = np.squeeze(img[:, :, i])

            # median, max and min
            sub_st = []
            sub_st += [np.mean(img_sub), np.std(img_sub), np.max(img_sub), np.median(img_sub), np.min(img_sub)]
            sub_st += [(sub_st[2] - sub_st[3]), (sub_st[2] - sub_st[4]), (sub_st[3] - sub_st[4])]
            sub_st += [(sub_st[-3] / sub_st[1]), (sub_st[-2] / sub_st[1]),
                       (sub_st[-1] / sub_st[1])]  # normalized by stdev
            st += sub_st
            # Laplacian, Sobel, kurtosis and skewness
            st_trans = []
            st_trans += [laplace(img_sub, mode='reflect', cval=0.0).ravel().var()]  # blurr
            sobel0 = sobel(img_sub, axis=0, mode='reflect', cval=0.0).ravel().var()
            sobel1 = sobel(img_sub, axis=1, mode='reflect', cval=0.0).ravel().var()
            st_trans += [sobel0, sobel1]
            st_trans += [kurtosis(img_sub.ravel()), skew(img_sub.ravel())]

            if opt_poly:
                st_interv.append(sub_st)
                #
                st += [x * y for x, y in combinations(st_trans, 2)]
                st += [x + y for x, y in combinations(st_trans, 2)]
                st += [x - y for x, y in combinations(st_trans, 2)]

                # hist
            # hist = list(cv2.calcHist([img], [i], None, [bins], [0., 1.]).flatten())
            hist = list(np.histogram(img_sub, bins=bins, range=(scl_min, scl_max))[0])
            hist_interv.append(hist)
            st += hist
            st += [hist.index(max(hist))]  # only the smallest index w/ max value would be incl
            st += [np.std(hist), np.max(hist), np.median(hist), (np.max(hist) - np.median(hist))]

        if opt_poly:
            for x, y in combinations(st_interv, 2):
                st += [float(x[j]) * float(y[j]) for j in range(len(st_interv[0]))]

        # correction
        nan = -999
        for i in range(len(st)):
            if isnan(st[i]) == True:
                st[i] = nan

    except:
        print('except: ')

    return [img_id, st]

def extract_img_stats(paths):
    imf_d = {}
    p = Pool(8) #(cpu_count())
    ret = p.map(img_to_stats, paths.items())
    for i in tqdm(range(len(ret)), miniters=100):
        imf_d[ret[i][0]] = ret[i][1]

    ret = []
    fdata = [imf_d[f] for f in paths.keys()]
    return np.array(fdata, dtype=np.float32)

def process(df, bands):

    data = extract_img_stats({k: v for k, v in zip(df['id'].tolist(), bands)})
    data = np.concatenate([data, df['inc_angle'].values[:, np.newaxis]], axis=-1)

    print(data.shape)
    return data

