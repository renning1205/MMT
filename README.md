## MMT: Mixed Mutual Transfer for Long-Tailed Image Classification
Ning Ren, Xiaosong Li, Yanxia Wu*, Yan Fu



## Main requirements

  * **torch == 1.0.1**
  * **torchvision == 0.2.2_post3**
  * **tensorboardX == 1.8**
  * **Python 3**

## Environmental settings
This repository is developed using python **3.5.2/3.6.7** on Ubuntu **16.04.5 LTS**. The CUDA nad CUDNN version is **9.0** and **7.1.3** respectively. For Cifar experiments, we use **one NVIDIA 1080ti GPU card** for training and testing. (**four cards for iNaturalist ones**). Other platforms or GPU cards are not fully tested.



## Usage
```bash
# To train long-tailed CIFAR-10 with imbalanced ratio of 50:
python main/train.py  --cfg configs/cifar10.yaml     

# To validate with the best model:
python main/valid.py  --cfg configs/cifar10.yaml

# To debug with CPU mode:
python main/train.py  --cfg configs/cifar10.yaml   CPU_MODE True
```

You can change the experimental setting by simply modifying the parameter in the yaml file.

## Data format

The annotation of a dataset is a dict consisting of two field: `annotations` and `num_classes`.
The field `annotations` is a list of dict with
`image_id`, `fpath`, `im_height`, `im_width` and `category_id`.

Here is an example.
```
{
    'annotations': [
                    {
                        'image_id': 1,
                        'fpath': '/home/MMT/CIFAIR-100-LT/images/train/7477/3b60c9486db1d2ee875f11a669fbde4a.jpg',
                        'im_height': 600,
                        'im_width': 800,
                        'category_id': 7477
                    },
                    ...
                   ]
    'num_classes': 8142
}
```


## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.

Ning.Ren.hrbeu@outlook.com

lixiaosong@hrbeu.edu.cn

wuyanxia@hrbeu.edu.cn

fuyan@hrbeu.edu.cn
