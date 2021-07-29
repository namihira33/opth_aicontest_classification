from trainer import Trainer

c = {
    'model_name': 'vgg16',
    'seed': [0,10,100,1000], 'bs': 64, 'lr': [1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1], 'n_epoch': [1]
}

trainer = Trainer(c)
trainer.run()