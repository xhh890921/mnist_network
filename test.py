from model import Network
from torchvision import transforms
from torchvision import datasets
import torch

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    # 读取测试数据集
    test_dataset = datasets.ImageFolder(root='./mnist_images/test', transform=transform)
    print("test_dataset length: ", len(test_dataset))

    model = Network()  # 定义神经网络模型
    model.load_state_dict(torch.load('mnist.pth')) # 加载刚刚训练好的模型文件

    right = 0 # 保存正确识别的数量
    for i, (x, y) in enumerate(test_dataset):
        output = model(x)  # 将其中的数据x输入到模型
        predict = output.argmax(1).item() # 选择概率最大标签的作为预测结果
        # 对比预测值predict和真实标签y
        if predict == y:
            right += 1
        else:
            # 将识别错误的样例打印了出来
            img_path = test_dataset.samples[i][0]
            print(f"wrong case: predict = {predict} y = {y} img_path = {img_path}")

    # 计算出测试效果
    sample_num = len(test_dataset)
    acc = right * 1.0 / sample_num
    print("test accuracy = %d / %d = %.3lf" % (right, sample_num, acc))
