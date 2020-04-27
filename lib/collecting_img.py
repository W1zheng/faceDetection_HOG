import lib.file_operate as fo
import lib.HOG as hog
import numpy as np
import cv2

'''
收集并提取样本的HOG特征，存储到文件
'''
def collecting_hog(pos_dir_path,output_path):
    file_list = []
    file_list = fo.find_all_picture(pos_dir_path)
    
    print("There are "+str(len(file_list))+" pictures in the folder.")
    
    count = 1     #处理样本计数器
    for pic_name in file_list:
        if count%100==0:
            print("processing pic "+str(count))
        count += 1 
    
        img = cv2.imread(pos_dir_path+'/'+pic_name,0)
        img = cv2.resize(img, (200,200),interpolation = cv2.INTER_AREA)
        hogDescriptor = hog.Hog_descriptor(img,cell_width=16, block_width=3, bin_size=8, block_stride=1,gamma=2.0)
        
        cur_hog_vec = hogDescriptor.extract()
        
        if cur_hog_vec is None:
            cur_hog_vec = []
        else:
            #cur_hog_vec = cur_hog_vec.ravel()
            cur_hog_vec = np.reshape(cur_hog_vec,[-1,5832])[0]
            fo.add_ndarray(cur_hog_vec,output_path)
            
        


        



