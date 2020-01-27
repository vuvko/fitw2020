import numpy as np
from pathlib import Path
import random
import logging
from itertools import combinations, starmap
import mxnet as mx
import cv2
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import Dataset

from typing import Tuple, Optional, Generator, Any, List, Union, Callable
import mytypes as t


class TestPairs(Dataset):

    def __init__(self, img_root: Path, csv_path: Path, save_ptype: bool = False, transform = None):
        super(TestPairs, self).__init__()
        self.img_root = img_root
        self._transform = transform
        self.pairs = []
        with open(str(csv_path), 'r') as f:
            f.readline()
            for line in f:
                idx, face1, face2, ptype = line.strip().split(',')
                self.pairs.append((self.img_root / face1, self.img_root / face2, ptype))
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[t.MxImg, t.MxImg]:
        face1_path, face2path, ptype = self.pairs[idx]
        face1 = mx.img.imread(str(face1_path))
        face2 = mx.img.imread(str(face2_path))
        if self._transform:
            face1 = self._transform(face1)
            face2 = self._transform(face2)
        return face1, face2, ptype


class RetrievalDataset(object):

    def __init__(self, gallery_root: Path, probe_root: Path, gallery_csv: Path, probe_csv: Path):
        super(RetrievalDataset, self).__init__()
        self.probes = []
        self.gallery = []
        with open(str(probe_csv), 'r') as f:
            f.readline()
            for line in f:
                idx, path = line.strip().split(',')
                self.probes.append((int(idx), probe_root / path))
        print('#Probes:', len(self.probes))
        with open(str(gallery_csv), 'r') as f:
            f.readline()
            for line in f:
                idx, path = line.strip().split(',')
                self.gallery.append((int(idx), gallery_root / path))
        print('#Gallery:', len(self.gallery))


class FamiliesDataset(Dataset):

    def __init__(self, families_root: Path, uniform_family: bool = False, transform = None):
        super(FamiliesDataset, self).__init__()
        self.families_root = families_root
        self._transform = transform
        whitelist_dir = 'MID'
        self.families = [[cur_person 
                          for cur_person in cur_family.iterdir() 
                              if cur_person.is_dir() and cur_person.name.startswith(whitelist_dir)] 
                         for cur_family in families_root.iterdir()]
        if uniform_family:
            # TODO: implement uniform distribution of persons per family
            raise NotImplementedError
        else:
            self.seq = [(img_path, family_idx, person_idx)
                        for family_idx, cur_family in enumerate(self.families)
                        for person_idx, cur_person in enumerate(cur_family)
                        for img_path in cur_person.iterdir()]
    
    def __getitem__(self, idx: int) -> Tuple[t.MxImg, int, int]:
       img_path, family_idx, person_idx = self.seq[idx]
       img = mx.img.imread(str(img_path))
       if self._transform is not None:
           img = self._transform(img)
       return img, family_idx, person_idx
    
    def __len__(self) -> int:
        return len(self.seq)


class PairDataset(Dataset):
    
    def __init__(self, 
                 families_dataset: FamiliesDataset, 
                 mining_strategy: str = 'random', 
                 num_pairs: Optional[int] = 10 ** 5):
        super(PairDataset, self).__init__()
        self.families = families_dataset
        if mining_strategy == 'random':
            if num_pairs > len(self.families) ** 2:
                logging.info(f'number of mined pairs is greater than number of all pairs')
            first_elems = random.choices(self.families.seq, k=num_pairs)
            second_elems = random.choices(self.families.seq, k=num_pairs)
            seq = zip(first_elems, second_elems)
            self.seq = [(img1, img2, int(family1 == family2))
                        for (img1, family1, _), (img2, family2, _) in seq]
        elif mining_strategy == 'all':
            seq = combinations(self.families.seq, 2)
            self.seq = [(img1, img2, int(family1 == family2))
                        for (img1, family1, _), (img2, family2, _) in seq]
        elif mining_strategy == 'balanced_random':
            assert num_pairs % 2 == 0
            num_positive = num_pairs // 2
            anchor_family_idx = np.random.uniform(size=(num_positive,))
            negative_family_idx = np.random.uniform(size=(num_positive,))
            anchor_family = []
            negative_family = []
            num_families = len(self.families.families)
            for cur_family_idx, negative_idx in zip(anchor_family_idx, negative_family_idx):
                family_idx = int(cur_family_idx * num_families)
                anchor_family.append(self.families.families[family_idx])
                negative_sample = self.families.families[:family_idx]
                if family_idx < num_families - 1:
                    negative_sample += self.families.families[family_idx+1:]
                negative_family.append(negative_sample[int(negative_idx * (num_families - 1))])
            triplets = list(starmap(self.mine_triplets, zip(anchor_family, negative_family)))
            positive_pairs = [(anchor, positive, 1) for anchor, positive, _ in triplets]
            negative_pairs = [(anchor, negative, 0) for anchor, _, negative in triplets]
            self.seq = positive_pairs + negative_pairs
        elif mining_strategy == 'balanced_hard':
            # TODO: implement balanced with hard negative mining strategy
            raise NotImplementedError
        else:
            logging.error(f'Uknown mining strategy {mining_strategy}')
            raise NotImplementedError
    
    @staticmethod
    def mine_triplets(anchor_family: List[Path],
                      negative_family: List[Path]) -> Tuple[Path, Path, Path]:
        
        def random_person_img(family: List[Path]) -> Tuple[int, Path]:
            idx = np.random.randint(0, len(family))
            person = family[idx]
            all_imgs = list(person.iterdir())
            img_path = random.choice(all_imgs)
            return idx, img_path
            
        anchor_idx, anchor = random_person_img(anchor_family)
        _, negative = random_person_img(negative_family)
        positive_family = anchor_family[:anchor_idx]
        if anchor_idx < len(anchor_family) - 1:
            positive_family += anchor_family[anchor_idx+1:]
        _, positive = random_person_img(positive_family)
        return anchor, positive, negative
    
    def __getitem__(self, idx: int) -> Tuple[t.MxImg, t.MxImg, int]:
        img1, img2, label = self.seq[idx]
        if not isinstance(img1, mx.nd.NDArray):
            img1 = mx.img.imread(str(img1)).astype(np.float32)
            img2 = mx.img.imread(str(img2)).astype(np.float32)
        return img1, img2, label
    
    def __len__(self) -> int:
        return len(self.seq)


class ImgDataset(Dataset):

    def __init__(self, paths: List[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> t.MxImg:
        return mx.img.imread(str(self.paths[idx])).transpose((2, 0, 1)).astype(np.float32)
