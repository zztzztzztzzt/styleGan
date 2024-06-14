import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
from torchsummary import summary
def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module
class FirstBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FirstBlock2d, self).__init__()
        kwargs = {'kernel_size': 7, 'stride': 1, 'padding': 3}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity)


    def forward(self, x):
        out = self.model(x)
        return out

class DownBlock2d(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(DownBlock2d, self).__init__()


        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)
        pool = nn.AvgPool2d(kernel_size=(2, 2))

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity, pool)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity, pool)

    def forward(self, x):
        out = self.model(x)
        return out
class FineEncoder(nn.Module):
    """docstring for Encoder"""
    def __init__(self, image_nc, ngf, img_f, layers, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineEncoder, self).__init__()
        self.layers = layers
        self.first = FirstBlock2d(image_nc, ngf, norm_layer, nonlinearity, use_spect)
        for i in range(layers):
            in_channels = min(ngf*(2**i), img_f)
            out_channels = min(ngf*(2**(i+1)), img_f)
            model = DownBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            setattr(self, 'down' + str(i), model)
        self.output_nc = out_channels

    def forward(self, x):

        x = self.first(x)
        out=[x]
        for i in range(self.layers):
            model = getattr(self, 'down'+str(i))
            x = torch.sigmoid(model(x))
            out.append(x)
        return out

# 该部分仅仅为了验证网络搭建的没问题
if __name__ == "__main__":
    # 定义一个设备
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    inx = torch.rand([1, 3, 512, 512])
    model = FineEncoder(3,32,512,7)  # 对于上面的类的实例化 并且放到设备里面
    # 显示一下模型每一层的参数数量
    x = model(inx)
    for i in range (8):
        print(x[i].shape)