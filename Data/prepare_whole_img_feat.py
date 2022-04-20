import torch
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.io as io
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import pickle
from multiprocessing import Pool

# load model
rn_50 = resnet50(pretrained=True)
rn_50.eval()
whole_img_extractor = create_feature_extractor(rn_50, {"avgpool":"feat"})

# prepare for file io
test_files = os.listdir("./test_images/")
test_img_feat = {}
train_files = os.listdir("./train_images/")
train_img_feat = {}

def feature_extraction(f, group):

    # load and resize image
    img = io.read_image("./{}_images/{}".format(group,f)) # c, w, h
    resized_img = F.interpolate(img.unsqueeze(0).type(torch.float),size=(224,224), mode="bilinear") # 1, c, 224, 224
    
    # check c
    if (resized_img.shape[1]==1): 
        resized_img = resized_img.repeat(1,3,1,1)
    
    # extract feature
    feat = whole_img_extractor(resized_img[:,:3,:,:])["feat"].view(2048)

    # save result
    d = {"whole_img_feat":feat, "img_c_w_h": img}
    np.save("./data/feat_whole_img/{}/{}.npy".format(group,f.split(".")[0]), d)
    #pickle.dump(d,open("./data/feat_whole_img/{}/{}.pkl".format(group,f.split(".")[0]), 'wb'))
    #torch.save(feat, "./data/feat_whole_img/{}/{}.npy".format(group,f.split(".")[0]))


def main():
    pool = Pool(os.cpu_count()-1) # 10 cpu available to use

    test_inputs = zip(test_files, ["test"]*len(test_files))
    train_inputs = zip(train_files, ["train"]*len(train_files))

    try:
        pool.starmap(feature_extraction, tqdm(test_inputs, total=len(test_files)))
        pool.starmap(feature_extraction, tqdm(train_inputs, total=len(train_files)))
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()
