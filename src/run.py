from trainer import Trainer
import sys
import numpy as np

seeds = list(np.random.choice(10000,3))

c = {
    'model_name': 'Resnet18','bs': 128,'n_epoch': 120,
    'lr': [7e-5,8e-5,9e-5,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4],'seed': seeds
}

args = len(sys.argv)
if args >= 2:
    c['model_name'] = sys.argv[1]
    c['cv'] = int(sys.argv[2].split('=')[1])
    c['evaluate'] = int(sys.argv[3].split('=')[1])

trainer = Trainer(c)
trainer.run()