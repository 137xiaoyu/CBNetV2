from albumentations import ShiftScaleRotate
import cv2
import matplotlib.pyplot as plt
from mmdet.datasets.pipelines import AutoAugment


def albu_visualize(img_name):
    src_img = cv2.imread(img_name)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    albu_augment = ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=0, p=1)

    dst_img = albu_augment(image=src_img)['image']
    print(f'src shape: {src_img.shape}\ndst shape: {dst_img.shape}')
    plt.figure(figsize=(10, 10))
    plt.imshow(dst_img)
    plt.show()


def autoaugment_visualize(img_name, policies):
    src_img = cv2.imread(img_name)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    auto_augment = AutoAugment(policies)
    
    results = {}
    results['img'] = src_img
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img']
    
    dst_img = auto_augment(results)['img']
    print(f'src shape: {src_img.shape}\ndst shape: {dst_img.shape}')
    plt.figure(figsize=(10, 10))
    plt.imshow(dst_img)
    plt.show()


if __name__ == '__main__':
    img_name = 'demo/demo.jpg'
    policies = [
        [
            dict(type='Resize',
                 img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                 multiscale_mode='value',
                 keep_ratio=True),
            dict(type='RandomCrop',
                 crop_type='absolute_range',
                 crop_size=(384, 600),
                 allow_negative_crop=True),
            dict(type='Resize',
                 img_scale=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                 multiscale_mode='value',
                 override=True,
                 keep_ratio=True)
        ]
    ]
    # albu_visualize(img_name)
    autoaugment_visualize(img_name, policies)
