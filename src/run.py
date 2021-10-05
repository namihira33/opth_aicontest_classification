from trainer import Trainer
import sys

c = {
    'model_name': 'Resnet18','n_epoch': list(range(1,2)),
    'seed': [0], 'bs': 64, 'lr': [5e-4]
}

args = len(sys.argv)
if args >= 2:
    c['model_name'] = sys.argv[1]

trainer = Trainer(c)
trainer.run()