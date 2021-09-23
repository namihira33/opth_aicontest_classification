from trainer import Trainer

c = {
    'model_name': ['Resnet18'],
    'seed': [0], 'bs': 64, 'lr': [1e-4], 'n_epoch': [100]
}

trainer = Trainer(c)
trainer.run()