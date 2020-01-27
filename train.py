import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms
from insightface.utils.face_align import norm_crop
from insightface import model_zoo
import time
import gluonfr
from pathlib import Path
import os
from tqdm import tqdm

from dataset import FamiliesDataset, UniformFamilySampler

import mytypes as t

class ReJPGTransform(gluon.nn.Block):

    def __init__(self, prob: float = 0.5, min_quality: int = 50):
        super(ReJPGTransform, self).__init__()
        self.prob = prob
        self.min_quality = min_quality

    def forward(self, x):
        if np.random.rand() < self.prob:
            quality = np.random.randint(self.min_quality, 99)
            img = x.asnumpy()
            _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            decoded = mx.img.imdecode(encoded).astype(np.float32)
            x = decoded
        return x


def train(normalize: bool = False):
    train_dataset = FamiliesDataset(Path('train-faces-det'))
    val_dataset = FamiliesDataset(Path('val-faces-det'))
    jitter_param = 0.15
    lighting_param = 0.15
    batch_size = 48
    num_workers = 12
    model_name = 'arcface_r100_v1'
    net_name = 'arcface_families_ft'
    weight_path = str(Path.home() / '.insightface/models/arcface_r100_v1/model-0000.params')
    sym_path = str(Path.home() / '.insightface/models/arcface_r100_v1/model-symbol.json')
    if normalize:
        snapshots_path = Path('snapshots_ft_norm')
    else:
        snapshots_path = Path('snapshots_ft')
    if not os.path.exists(str(snapshots_path)):
        os.makedirs(str(snapshots_path))
    num_families = len(train_dataset.families)

    warmup = 200
    lr = 1e-4
    cooldown = 400
    lr_factor = 0.75
    num_epoch = 20
    momentum = 0.9
    wd = 1e-4
    clip_gradient = 1.0
    lr_steps = [8, 14, 25, 35, 40, 50, 60]
    ctx_list = [mx.gpu(0)]

    transform_img_train = transforms.Compose([
            transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                         saturation=jitter_param),
            transforms.RandomLighting(lighting_param),
            # ReJPGTransform(0.3, 70),
            transforms.ToTensor()
        ])

    transform_img_val = transforms.Compose([
            transforms.ToTensor()
        ])
    train_data = mx.gluon.data.DataLoader(
            train_dataset.transform_first(transform_img_train),
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
    val_data = mx.gluon.data.DataLoader(
            train_dataset.transform_first(transform_img_val),
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
    ctx = ctx_list[0]
    sym = mx.sym.load(str(sym_path))
    if normalize:
        norm_sym = mx.sym.sqrt(mx.sym.sum(sym ** 2, axis=1, keepdims=True) + 1e-6)
        sym = mx.sym.broadcast_div(sym, norm_sym, name='fc_normed') * 32
    sym = mx.sym.FullyConnected(sym, num_hidden=num_families, name='fc_classification', lr_mult=1)
    net = gluon.SymbolBlock([sym], [mx.sym.var('data')])
    net.load_parameters(str(weight_path), ctx=mx.cpu(), cast_dtype=True,
                        allow_missing=True, ignore_extra=False)
    net.initialize(mx.init.Normal(), ctx=mx.cpu())
    net.collect_params().reset_ctx(ctx)
    net.hybridize()

    all_losses = [
                  ('softmax', gluon.loss.SoftmaxCrossEntropyLoss()),
                  # ('arc', gluonfr.loss.ArcLoss(num_families, m=0.7, s=32, easy_margin=False)), 
                  # ('center', gluonfr.loss.CenterLoss(num_families, 512, 1e-1))
                  ]
    
    ##############
    
    if warmup > 0:
        start_lr = 1e-10
    else:
        start_lr = lr
    warmup_iter = 0
    end_iter = num_epoch * len(train_data)
    cooldown_start = end_iter - cooldown
    cooldown_iter = 0
    end_lr = 1e-10
    param_dict = net.collect_params()
    trainer = mx.gluon.Trainer(param_dict, 'sgd', {
        'learning_rate': start_lr, 'momentum': momentum, 'wd': wd, 'clip_gradient': clip_gradient})

    lr_counter = 0
    num_batch = len(train_data)

    for epoch in range(num_epoch):
        if epoch == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate*lr_factor)
            lr_counter += 1

        tic = time.time()
        losses = [0] * len(all_losses)
        metric = mx.metric.Accuracy()
        print(' > training', epoch)
        for i, batch in tqdm(enumerate(train_data), total=len(train_data)):
            if warmup_iter < warmup:
                cur_lr = (warmup_iter + 1) * (lr - start_lr) / warmup + start_lr
                trainer.set_learning_rate(cur_lr)
                warmup_iter += 1
            elif cooldown_iter > cooldown_start:
                cur_lr = (end_iter - cooldown_iter) * (trainer.learning_rate - end_lr) / cooldown + end_lr
                trainer.set_learning_rate(cur_lr)
            cooldown_iter += 1
            data = mx.gluon.utils.split_and_load(batch[0] * 255, ctx_list=ctx_list, even_split=False)
            gts = mx.gluon.utils.split_and_load(batch[1], ctx_list=ctx_list, even_split=False)
            with ag.record():
                outputs = [net(X) for X in data]
                if np.any([np.any(np.isnan(o.asnumpy())) for os in outputs for o in os]):
                    print('OOps!')
                    raise RuntimeError
                cur_losses = [[cur_loss(o, l) for (o, l) in zip(outputs, gts)] for _, cur_loss in all_losses]
                metric.update(gts, outputs)
                combined_losses = [cur[0] for cur in zip(*cur_losses)]
                if np.any([np.any(np.isnan(l.asnumpy())) for l in cur_losses[0]]):
                    print('OOps2!')
                    raise RuntimeError
            for combined_loss in combined_losses:
                combined_loss.backward()

            trainer.step(batch_size, ignore_stale_grad=True)
            for idx, cur_loss in enumerate(cur_losses):
                losses[idx] += sum([l.mean().asscalar() for l in cur_loss]) / len(cur_loss)

        if (epoch + 1) % 10 == 0:
            net.save_parameters(str(snapshots_path / f'{net_name}-{(epoch + 1):04d}.params'))

        losses = [l / num_batch for l in losses]
        losses_str = [f'{l_name}: {losses[idx]:.3f}' for idx, (l_name, _) in enumerate(all_losses)]
        losses_str = '; '.join(losses_str)
        m_name, m_val = metric.get()
        losses_str += f'| {m_name}: {m_val}'
        print(f'[Epoch {epoch:03d}] {losses_str} | time: {time.time() - tic:.1f}')


if __name__ == '__main__':
    np.random.seed(100)
    mx.random.seed(100)
    train(normalize=False)
    # train(normalize=True)
