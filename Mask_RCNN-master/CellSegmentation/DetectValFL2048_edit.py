import os
import sys
import numpy as np
import pickle
import skimage.measure
import skimage.io
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed


ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
# Root directory of the project
def search_overlap_labimg(image_name_val, obj1, ROIsize):
    with open(image_name_val, 'rb') as k:
        image_mask1 = pickle.load(k)
    try:
         # Numofobj = image_mask1.shape[2]
        Numofobj = len(image_mask1)
    except:
        print(image_name_val)
    overlap_size_list = []
    for j in range(0, Numofobj):
        mask = image_mask1[j]
        obj2 = np.zeros((ROIsize, ROIsize), dtype=int)
        coor2 = mask
        obj2[coor2] = 1
        obj_overlap = obj2 + obj1
        area_overlap = np.where(obj_overlap == 2)
        overlap_size_list.append(len(area_overlap[0]))
    maxsize = max(overlap_size_list)
    idx = overlap_size_list.index(maxsize)
    obj_overlap = np.zeros((ROIsize, ROIsize), dtype=int)
    coor_test = image_mask1[idx]
    obj_overlap[coor_test] = 1
    return obj_overlap, coor_test, maxsize

def flatMasks(filename_Seg, ROIsize):
    with open(filename_Seg, 'rb') as k:
        image_mask_seg = pickle.load(k)
    try:
        Numofobj = image_mask_seg.shape[2]
    except:
        print(filename_Seg)
    I_mask = np.zeros((ROIsize, ROIsize), dtype=int)
    for idx in image_mask_seg:
        I_mask[idx] = 1
    # image_mask_seg.astype(int)
    # img_mask = np.sum(image_mask_seg, axis=2)
    # idx = np.where(img_mask!=0)
    # I_mask = np.zeros((img_mask.shape[0],img_mask.shape[1]), dtype=int)
    # I_mask[idx] = 1
    # plt.imshow(I_mask)
    # plt.show()
    return I_mask

def compute_instance_EM(image_mask1, i, image_name_val, ROIsize):
    # object = image_mask1[:, :, i]
    obj1 = np.zeros((ROIsize, ROIsize), dtype=int)
    coor1 = image_mask1[i]
    obj1[coor1] = 1
    obj_overlap, coor_test, max_overlapsize = search_overlap_labimg(image_name_val, obj1, ROIsize)
    obj1 = obj1.flatten()
    obj_overlap = obj_overlap.flatten()
    f1 = f1_score(obj1, obj_overlap)
    kappa = cohen_kappa_score(obj1, obj_overlap)
    # overlap_percentage = max_overlapsize/(len(coor1[0])+len(coor_test[0]))   # compute IoU
    # overlap_percent_G.append(overlap_percentage)
    kappa_weighted = kappa * len(coor1[0])
    f1_weighted = f1 * len(coor1[0])
    obj_area = len(coor1[0])
    if f1 <= 0.000001:          # when f1 equals to 0, where there is an FN, using the maximum axis lenghth as housdorff score.
        # print(overlap_percentage)
        img_temp = np.zeros((ROIsize, ROIsize), dtype=int)
        img_temp[coor1] = 1
        props_temp = skimage.measure.regionprops(img_temp)
        obj_temp = props_temp[0]
        h_score = obj_temp.major_axis_length
        # FP += 1
        # Img_FPs[coor1] = 1
        h_weighted = h_score * len(coor1[0])
        # plt.imshow(img)
        # plt.show()
    else:
        a = np.column_stack((coor_test[0], coor_test[1]))
        b = np.column_stack((coor1[0], coor1[1]))
        h_score = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])
        h_weighted = h_score * len(coor1[0])
    Dict_EM = {'f1_w':f1_weighted, 'K_w':kappa_weighted, 'h_w':h_weighted, 'obj_a': obj_area}
    # return f1_weighted, kappa_weighted, h_weighted, obj_area
    return Dict_EM

def loop_objects_val_objects(image_seg, image_name_val, ROIsize):
    with open(image_seg, 'rb') as k:
        image_mask1 = pickle.load(k)
    Numofobj = len(image_mask1)
    inputs = tqdm(range(0, Numofobj))
    num_cores = multiprocessing.cpu_count()
    Dict_EM = Parallel(n_jobs=num_cores)(delayed(compute_instance_EM)(image_mask1, i, image_name_val, ROIsize) for i in inputs)
    return Dict_EM

def pharsDictResult(Dict_EM):
    f_g=[]
    k_g=[]
    h_g=[]
    obj_area=[]
    for Dict in Dict_EM:
        f1_w = Dict['f1_w']
        k1_w = Dict['K_w']
        h1_w = Dict['h_w']
        obj = Dict['obj_a']
        f_g.append(f1_w)
        k_g.append(k1_w)
        h_g.append(h1_w)
        obj_area.append(obj)
    return f_g, k_g, h_g, obj_area

def computeErrorMetricsMask(ExpName, Path_Seg, Path_GT, filename, ROIsize):
    # compute error metrics for one round
    ID = filename.replace('_'+ExpName, '')
    filename_Seg = os.path.join(Path_Seg, filename)
    filename_GT = os.path.join(Path_GT,ID)
    I_seg_mask = flatMasks(filename_Seg, ROIsize)
    I_GT_mask = flatMasks(filename_GT, ROIsize)
    obj1 = I_seg_mask.flatten()
    obj_overlap = I_GT_mask.flatten()
    # f1_pixel = f1_score(obj1, obj_overlap)
    # ac_pixel = accuracy_score(obj1, obj_overlap)

    # compute object-error metrics using segmentation results as reference
    Dict_EM1 = loop_objects_val_objects(filename_Seg, filename_GT, ROIsize)

    # compute object-error metrics using manual annotation as reference
    Dict_EM2= loop_objects_val_objects(filename_GT, filename_Seg, ROIsize)
    print('finish parallel computing')

    f1_g1, kappa_g1, h_g1, Object_area1 = pharsDictResult(Dict_EM1)
    f1_g2, kappa_g2, h_g2, Object_area2 = pharsDictResult(Dict_EM2)
    f1 = (sum(f1_g1) / sum(Object_area1) + sum(f1_g2) / sum(Object_area2)) / 2
    # kappa = (sum(kappa_g1) / sum(Object_area1) + sum(kappa_g2) / sum(Object_area2)) / 2
    h = (sum(h_g1) / sum(Object_area1) + sum(h_g2) / sum(Object_area2)) / 2
    EM = [f1, h]

    return EM

"""setups for inferencing:
1)result file save path
2)result directory"""
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for cell counting and segmentation')
    parser.add_argument('--yaml', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run validation on")
    args = parser.parse_args()
    import yaml
    yaml_filename = args.yaml
    with open(yaml_filename, 'r') as file:
        dic1 = yaml.full_load(file)
    args.dataset = dic1['input']
    args.ExpName = dic1['ExpName']
    ROOT_DIR = os.path.abspath("..")
    print(ROOT_DIR)
    outputpath = dic1['output']
    RESULTS_DIR = os.path.join(ROOT_DIR, outputpath)
    print(RESULTS_DIR)
# Dataset directory #
ExpName = args.ExpName
ROIsize = 2048
path_current = os.path.dirname(os.path.realpath(__file__))
Path_GT = os.path.join(args.dataset, 'GTPyIdx')
if not os.path.exists(Path_GT):
    Path_GT = os.path.join(ROOT_DIR, args.dataset, 'GTPyIdx')

## validate the segmentation resutls against GT ##
Path_Seg = os.path.join(RESULTS_DIR, 'infer_results')
val_path =os.path.join(RESULTS_DIR, 'ValResults')
if os.path.isdir(val_path)==False:
    os.makedirs(val_path)
EM_result_path = os.path.join(val_path, ExpName)
if os.path.isdir(EM_result_path) == False:
    os.makedirs(EM_result_path)

SegFileList = os.listdir(Path_Seg)
image_names = sorted(os.listdir(Path_Seg))
# num_cores = multiprocessing.cpu_count()
# print('number of cores', num_cores)

EM = []
for i in image_names:
    EM_sub = computeErrorMetricsMask(ExpName, Path_Seg, Path_GT, i, ROIsize)
    EM.append(EM_sub)
    filename = ExpName + '_' + str(i)
    savefilename = os.path.join(EM_result_path, filename)
    with open(savefilename, 'wb') as j:
        pickle.dump(EM_sub, j)
    print('finish one WSI')

EM_G = np.zeros((len(EM), 2))
counter = 0
for E in EM:
    EM_arr = np.asarray(E)
    EM_G[counter, :] = EM_arr
    counter = counter+1
EM_mean = np.mean(EM_G, axis=0)
EM_std = np.std(EM_G, axis=0)
print('mean', 'OD', 'OH')
print(EM_mean)
print('standard deviation', 'OD', 'OH')
print(EM_std)