import os
import sys
import glob
import h5py
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(os.path.curdir))
    DATA_DIR = os.path.join(BASE_DIR, 'ShapeNet')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'complete')):
        #!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=154C0HWNXmQhIavytPLq-MwNzg46OaLBc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=154C0HWNXmQhIavytPLq-MwNzg46OaLBc" -O complete.zip && rm -rf /tmp/cookies.txt  

        path=DATA_DIR +'/complete.zip'
        zipfile = os.path.basename(path)
        os.system('unzip %s' % (zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        #os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(os.path.curdir))
    DATA_DIR = os.path.join(BASE_DIR, 'ShapeNet')
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR +'/complete', "[!README]*"))
    for i, folder in enumerate(folders):
      print("processing class: {}".format(os.path.basename(folder)))
      # store folder name with ID so we can retrieve later
      class_map[i] = folder.split("/")[-1]
      # gather all files
      train_files = glob.glob(os.path.join(folder, "train/*"))
      test_files = glob.glob(os.path.join(folder, "test/*"))

      for f in train_files:
          train_pcd=o3d.io.read_point_cloud(f)
          train_points.append(train_pcd.points)
          train_labels.append(i)

      for f in test_files:
          test_pcd=o3d.io.read_point_cloud(f)
          test_points.append(test_pcd.points)
          test_labels.append(i)
    ### subsampling the pointcloud and reducing it's size to one eighth of the original size ###      
    train_sub=np.empty((train_points.shape[0],2048,3))
    test_sub=np.empty((test_points.shape[0],2048,3))
    train_slable=np.empty((train_label.shape[0]))
    test_slabel=np.empty((train_label.shape[0]))
    for i in range(train_points.shape[0]):
        train_sub[i]=train_points[i][::8]
        test_sub[i]=test_points[i][::8]
        train_slable[i]=train_labels[::8]
        test_slabel[i]=test_labels[::8]
        


    train_points=np.array(train_sub)
    test_points=np.array(test_points_sub)
    train_labels=np.array(train_slabel)
    test_labels=np.array(test_slabel)
    class_map
    np.save('train_points.npy',train_points)
    np.save('train_labels.npy',train_labels)
    return train_points, test_points, train_labels, test_labels


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    np.save('translated_pointcloud.npy',translated_pointcloud)
    
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    np.save('pointcloud.npy',pointcloud)
    return pointcloud


class ShapeNet(Dataset):
    def __init__(self, num_points, partition='train', gaussian_noise=False, unseen=False, factor=4):
        self.data, self.test_data, self.label, self.test_label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.test_label = self.test_label.squeeze()
        self.factor = factor
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.test_data
                self.label = self.test_label
            elif self.partition == 'train':
                self.data = self.train_data
                self.label = self.train_label
            self.

    def __getitem__(self, item):
        
        pointcloud = self.data[item][:self.num_points]
        p
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        np.save('translation_ab.npy',translation_ab)
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        np.save('rotation_ab.npy',rotation_ab)
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T
        np.save('pointcloud1.npy',pointcloud1)
        np.save('pointcloud2.npy',pointcloud2)

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ShapeNet(1048)
    test = ShapeNet(1048, 'test')
    from numpy import savetxt
    #np.savetxt('train.csv',np.array(train),delimiter=",")
    #np.savetxt('test.csv',np.array(test),delimiter=",")
    for data in train:
        print(len(data))
        break
