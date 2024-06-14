
**模型权重下载在--checkpoint下**
[--checkpoint](https://pan.baidu.com/s/1xrMIL1MCE7l7QvF8OF0m_g?pwd=ww99)
**提取码：ww99**<br>
**主要修改的文件--model--stylegan--model.py;--finetune_stylegan.py**


运行
> python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_stylegan.py --iter 4000
                          --batch 2 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style cartoon
                          

