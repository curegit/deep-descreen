import os
import chainer
from chainer import optimizers
from chainer.training import Trainer, extensions
from chainer.iterators import MultithreadIterator
from chainer.serializers import save_hdf5
from dataset import ImageArrayDataset
from updater import CustomUpdater
from network import Network

batch = 16
epoch = 1000
channels = 64
blocks = 2
tblocks = 4
ksize = 3
tksize = 3
device = 0
patch = 512
out = "result"

chainer.global_config.train = True
chainer.global_config.autotune = True
chainer.global_config.cudnn_deterministic = False

os.makedirs(out, exist_ok=True)

training = ImageArrayDataset("dataset")
validation = ImageArrayDataset("validation")
t_iter = MultithreadIterator(training, batch_size=batch, repeat=True, shuffle=True, n_threads=batch)
v_iter = MultithreadIterator(validation, batch_size=batch, repeat=True, shuffle=True, n_threads=batch)

model = Network(channels, blocks, ksize, tblocks, tksize)
if device >= 0: model.to_gpu()
optimizer = optimizers.Adam().setup(model)
updater = CustomUpdater({"main": t_iter, "test": v_iter}, optimizer, (patch, patch))

trainer = Trainer(updater, (epoch, "epoch"), out=out)
log = extensions.LogReport()
trainer.extend(log)
trainer.extend(extensions.PrintReport(["epoch", "iteration", "loss", "test"], log))
trainer.extend(extensions.ProgressBar(update_interval=1))
trainer.extend(lambda trainer: save_hdf5(f"{out}/m{trainer.updater.iteration}.hdf5", model), trigger=(5, "epoch"))
trainer.extend(lambda trainer: save_hdf5(f"{out}/o{trainer.updater.iteration}.hdf5", optimizer), trigger=(5, "epoch"))

trainer.run()
save_hdf5(f"{out}/model.hdf5", model)
save_hdf5(f"{out}/optimizer.hdf5", optimizer)
