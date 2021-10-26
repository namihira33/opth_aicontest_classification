from trainer import Trainer
import sys
import numpy as np

seeds = list(np.random.choice(10000,1))

c = {
    'model_name': 'Resnet34','bs': 64,'n_epoch': 10000,
    'lr': [8e-5],'seed': seeds,'p':0.25
}

args = len(sys.argv)
if args >= 2:
    c['model_name'] = sys.argv[1]
    c['cv'] = int(sys.argv[2].split('=')[1])
    c['evaluate'] = int(sys.argv[3].split('=')[1])

trainer = Trainer(c)
trainer.run()