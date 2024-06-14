import argparse
import math
import random
import os
from PIL import Image
import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from util import data_sampler, requires_grad, accumulate, sample_data, d_logistic_loss, d_r1_loss, g_nonsaturating_loss, \
    g_path_regularize, make_noise, mixing_noise, set_grad_none, Landmark_loss, gram
from model.encoder.criteria import id_loss
import lpips
from vgg import Vgg16
import dlib
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


try:
    import wandb

except ImportError:
    wandb = None

def k_split_images(image_paths, k):
    images = image_paths
    width, height = images[0].size
    cropped_width = width//k
    cropped_height = height

    result_image = Image.new('RGB', (cropped_width * 4, height * k))

    for i in range(k):
        for j, image in enumerate(images):

            cropped_image = image.crop((cropped_width*i, 0, (i+1)*cropped_width, cropped_height))

            result_image.paste(cropped_image, (j * cropped_width, i * cropped_height))

    return result_image



from model.stylegan.dataset import FramesDataset
from model.stylegan.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from model.stylegan.non_leaking import augment, AdaptiveAugment
from model.stylegan.model import Generator, Discriminator, ResNet, Bottleneck


class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Train StyleGAN")
        self.parser.add_argument("--path_D", type=str,
                                 default='/home/featurize/DualStyleGAN/data/SCUT-FBP5500_v2/Imagesd')
        self.parser.add_argument("--path_S", type=str,
                                 default='/home/featurize/DualStyleGAN/data/SCUT-FBP5500_v2/imagess')
        self.parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
        self.parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
        self.parser.add_argument("--n_sample", type=int, default=16,
                                 help="number of the samples generated during training")
        self.parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
        self.parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        self.parser.add_argument("--path_regularize", type=float, default=2,
                                 help="weight of the path length regularization")
        self.parser.add_argument("--path_batch_shrink", type=int, default=2,
                                 help="batch size reducing factor for the path length regularization (reduce memory consumption)")
        self.parser.add_argument("--d_reg_every", type=int, default=16,
                                 help="interval of the applying r1 regularization")
        self.parser.add_argument("--g_reg_every", type=int, default=4,
                                 help="interval of the applying path length regularization")
        self.parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
        self.parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
        self.parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
        self.parser.add_argument("--channel_multiplier", type=int, default=2,
                                 help="channel multiplier factor for the model. config-f = 2, else = 1")
        self.parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
        self.parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        self.parser.add_argument("--augment", action="store_true", default=None, help="apply non leaking augmentation")
        self.parser.add_argument("--augment_p", type=float, default=0,
                                 help="probability of applying augmentation. 0 = use adaptive augmentation")
        self.parser.add_argument("--ada_target", type=float, default=0.6,
                                 help="target augmentation probability for adaptive augmentation")
        self.parser.add_argument("--ada_length", type=int, default=500 * 1000,
                                 help="target duraing to reach augmentation probability for adaptive augmentation")
        self.parser.add_argument("--ada_every", type=int, default=256,
                                 help="probability update interval of the adaptive augmentation")
        self.parser.add_argument("--save_every", type=int, default=10000, help="interval of saving a checkpoint")
        self.parser.add_argument("--style", type=str, default='cartoon', help="style type")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path to save the model")

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        if self.opt.local_rank == 0:
            print('Load options')
            for name, value in sorted(args.items()):
                print('%s: %s' % (str(name), str(value)))
        return self.opt


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema,id_loss, device):
    loader = sample_data(loader)
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, ncols=140, dynamic_ncols=False, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    land_loss = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")

            break
        #从文件读取两个相同id的drive和source和两个不同id的drive和source,两个drive的landmark和对应的关键点
        real_img_D, real_img_S, real_img_D_img_pt ,pt1,real_img_D_noid, real_img_S_noid, real_img_D_img_pt_noid ,pt2= next(loader)
        pt_D = torch.cat([pt1, pt2], dim=0)
        real_img_D = torch.cat([real_img_D, real_img_D_noid], dim=0)#drive前面的是相同id后面是不同id
        real_img_S = torch.cat([real_img_S, real_img_S_noid], dim=0)#source前面的是相同id后面是不同id
        real_img_D_img_pt = torch.cat([real_img_D_img_pt, real_img_D_img_pt_noid], dim=0)#drive的landmark
        
        real_img_D = real_img_D.to(device)
        real_img_S = real_img_S.to(device)
        real_img_D_img_pt = real_img_D_img_pt.to(device)

        real_img = real_img_S

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        fake_img, _ = generator(real_img_D, real_img_S, real_img_D_img_pt)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)
        loss_dict["d"] = d_loss

        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)
        dtype = torch.cuda.FloatTensor
        
        vgg = Vgg16().type(dtype)
        requires_grad(vgg, False)
        
        fake_img, _ = generator(real_img_D, real_img_S, real_img_D_img_pt.to(device))

        #percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))

        # 按照batch维度平均分割张量
        fake_img_id, fake_img_noid = torch.split(fake_img,args.batch, dim=0)
        real_img_D_id, real_img_D_noid = torch.split(real_img_D, args.batch, dim=0)
        real_img_S_id, real_img_S_noid = torch.split(real_img, args.batch, dim=0)
        #查看是否有梯度############
        loss_mse = torch.nn.MSELoss()

        #lp_loss和drive做 l1也是drive
        y_c_features = vgg(real_img_D_id.type(dtype))#和drive做修改
        y_hat_features = vgg(fake_img_id.type(dtype))

        recon = y_c_features[1]
        recon_hat = y_hat_features[1]
        content_loss = loss_mse(recon_hat, recon)
        #l1_loss
        l1_loss_function = nn.L1Loss()
        l1_loss = l1_loss_function(fake_img_id, real_img_D_id)
        #id_loss
        ID_loss = (torch.tensor(0.0).to(device) if 1 == 0 else id_loss(fake_img_id, real_img_S_id) * 1)
        #land_mark_loss
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('/home/featurize/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat')
        tensor_list = torch.unbind(fake_img, dim=0)
        land_loss = 0
        # 打印拆分后的每个张量
        for g, split_tensor in enumerate(tensor_list):
            gray = cv2.cvtColor(np.array(split_tensor.detach().cpu().numpy()).transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
            faces = detector(np.uint8(gray))
            if faces:
                landmarks = predictor(np.uint8(gray), faces[0])
                crop_pts = []
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    crop_pts.append(int(x))
                    crop_pts.append(int(y))
                land_loss += Landmark_loss(pt_D,crop_pts)
        
        fake_pred = discriminator(fake_img)
        p_loss = content_loss #+ style_loss
        g_loss = g_nonsaturating_loss(fake_pred) + ID_loss + p_loss + land_loss + l1_loss
        #g_loss = g_nonsaturating_loss(fake_pred)
        loss_dict["g"] = g_loss
        loss_dict["lp"] = p_loss
        loss_dict["id"] = ID_loss
        loss_dict["l1"] = l1_loss
        loss_dict["landmark"] = land_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        id_loss_val = loss_reduced["id"].mean().item()
        lp_loss = loss_reduced["lp"].mean().item()
        l1_loss_val = loss_reduced["l1"].mean().item()
        landmark = loss_reduced["landmark"]
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"iter: {i:05d}; d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    # f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f};"
                    f"land: {landmark:.4f};"
                    f"lp: {lp_loss:.4f};"
                    f"id: {id_loss_val:.4f};"
                    f"l1: {l1_loss_val:.4f};"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        #"Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        #"Path Length": path_length_val,
                    }
                )
            if i % 100 == 0 or (i + 1) == args.iter:###需要看是不是【-1，1】#s，land,fake,d
                with torch.no_grad():
                    g_ema.eval()
                    sample, pt = g_ema(real_img_D, real_img_S, real_img_D_img_pt)
                    sample = F.interpolate(sample, 256)
                    real_img_D = F.interpolate(real_img_D, 256)
                    real_img_S = F.interpolate(real_img_S, 256)
                    real_img_D_img_pt = F.interpolate(real_img_D_img_pt, 256)
                    utils.save_image(
                        sample,
                        f"log/%s/finetune-%06d.jpg" % (args.style, i),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(
                        real_img_D,
                        f"log/%s/finetuneD-%06d.jpg" % (args.style, i),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(
                        real_img_S,
                        f"log/%s/finetuneS-%06d.jpg" % (args.style, i),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(
                        real_img_D_img_pt,
                        f"log/%s/finetuneDpt-%06d.jpg" % (args.style, i),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    img1 = Image.open(f"log/%s/finetune-%06d.jpg" % (args.style, i))
                    img2 = Image.open(f"log/%s/finetuneD-%06d.jpg" % (args.style, i))
                    img3 = Image.open(f"log/%s/finetuneS-%06d.jpg" % (args.style, i))
                    img4 = Image.open(f"log/%s/finetuneDpt-%06d.jpg" % (args.style, i))
                    image_paths = [img1, img2, img3, img4]
                    k = 2
                    result = k_split_images(image_paths, k)
                    result.save(f"log/%s/finetunenew-%06d.jpg" % (args.style, i))
            if (i + 1) % args.save_every == 0 or (i + 1) == args.iter:
                torch.save(
                    {
                        # "g": g_module.state_dict(),
                        # "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        # "g_optim": g_optim.state_dict(),
                        # "d_optim": d_optim.state_dict(),
                        # "args": args,
                        # "ada_aug_p": ada_aug_p,
                    },
                    f"%s/%s/finetune-%06d.pt" % (args.model_path, args.style, i + 1),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = TrainOptions()
    args = parser.parse()
    if args.local_rank == 0:
        print('*' * 98)

    if not os.path.exists("log/%s/" % (args.style)):
        os.makedirs("log/%s/" % (args.style))
    if not os.path.exists("%s/%s/" % (args.model_path, args.style)):
        os.makedirs("%s/%s/" % (args.model_path, args.style))

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    # if args.arch == 'stylegan2':
    # from model.stylegan.model import Generator, Discriminator

    # elif args.arch == 'swagan':
    # from swagan import Generator, Discriminator

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()

    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        # ckptres50 = torch.load('./checkpoint/model_ir_se50.pth', map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"], strict=False)
        #discriminator.load_state_dict(ckpt["d"], strict=False)#不要加载判别器
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)
        if "g_optim" in ckpt:
            g_optim.load_state_dict(ckpt["g_optim"])
        if "d_optim" in ckpt:
            d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    #id_loss = id_loss.IDLoss(os.path.join("/home/featurize/DualStyleGAN/checkpoint/", 'model_ir_se50.pth')).to(
        #device).eval()

    dataset = FramesDataset(root_dir=r"/home/featurize/DualStyleGAN/data/vox1_1s/vox1_s")
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")
    id_loss = id_loss.IDLoss('./checkpoint/model_ir_se50.pth').to(device).eval()
    requires_grad(id_loss, False)
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, id_loss,device)
