import ProtoNet
import torch
import cv2
import os
from torchvision import transforms
PATH = 'Data/recogition/pic/mahjong'
DATA0_PATH = 'Data/recogition/data0'


class ImgDetectorDL:
    def __init__(self):
        self.device = 'cpu'
        self.model = ProtoNet.get_few_shot_encoder().to(self.device)
        self.model.load_state_dict(torch.load('model3.pth'))
        self.model.eval()
        self.sp_model = ProtoNet.get_few_shot_encoder().to(self.device)
        self.sp_model.load_state_dict(torch.load('19w_model.pth'))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((40, 40)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 计算proto
        self.prototypes = {}
        self._calculate_prototypes(self.model, self.prototypes)
        # 计算19w proto
        self.sp_prototypes = {}
        self._calculate_prototypes(self.sp_model, self.sp_prototypes, 'm')

    def detect(self, img):
        min_key, min_d = self._detect(img, self.model, self.prototypes)
        # TODO: compare min_d to certain the tile is W
        if 'm' in min_key:
            min_key, min_d = self._detect(img, self.sp_model, self.sp_prototypes)
        return min_key

    def _detect(self, img, model, prototypes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y_pred = model(self.transform(img).unsqueeze(0).to(self.device))
        min_d = None
        min_key = None
        for key in prototypes.keys():
            distance = ProtoNet.pairwise_distances(prototypes[key], y_pred, 'l2')
            if min_d is None or min_d > distance:
                min_d = distance
                min_key = key
        return min_key, min_d

    def _calculate_prototypes(self, model, dic, special=''):
        for _class in os.listdir(DATA0_PATH):
            path2 = DATA0_PATH + '/' + _class
            y_pred = None
            count = 1
            if special != '' and not (special in _class):
                continue

            for imgf in os.listdir(path2):
                path3 = path2 + '/' + imgf
                img = cv2.imread(path3)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if y_pred is None:
                    y_pred = model(self.transform(img).unsqueeze(0).to(self.device))
                else:
                    y_pred = y_pred / count * (count - 1)
                    y_pred += model(self.transform(img).unsqueeze(0).to(self.device)) / count
                count += 1
            dic[_class] = y_pred
            if count == 1:
                raise IOError("缺少样本图片")


if __name__ == '__main__':
    detector = ImgDetectorDL()
    print(detector.detect(cv2.imread(PATH+'/1p.png')))
    print(detector.detect(cv2.imread(PATH+'/2m.png')))
    print(detector.detect(cv2.imread(PATH+'/3z.png')))
    print(detector.detect(cv2.imread(PATH+'/9m.png')))
    print(detector.detect(cv2.imread(PATH+'/5z.png')))