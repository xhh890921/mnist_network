# 这个程序的功能会先将MNIST数据下载下来，然后再保存为.png的格式。

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# 注意，这里得到的train_data和test_data已经直接可以用于训练了！
# 不一定要继续后面的保存图像。
train_data = MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_data = MNIST(root='./data', train=False, download=True, transform=ToTensor())

# 继续将手写数字图像“保存”下来
# 输出两个文件夹，train和test，分别保存训练和测试数据。
# train和test文件夹中，分别有0、1、2、3、4、5、6、7、8、9；
# 这10个子文件夹保存每种数字的数据

from torchvision.transforms import ToPILImage

train_data = [(ToPILImage()(img), label) for img, label in train_data]
test_data = [(ToPILImage()(img), label) for img, label in test_data]

import os
import secrets
def save_images(dataset, folder_name):
    root_dir = os.path.join('./mnist_images', folder_name)
    os.makedirs(root_dir, exist_ok=True)
    for i in range(len(dataset)):
        img, label = dataset[i]
        label_dir = os.path.join(root_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        random_filename = secrets.token_hex(8) + '.png'
        img.save(os.path.join(label_dir, random_filename))

save_images(train_data, 'train')
save_images(test_data, 'test')






