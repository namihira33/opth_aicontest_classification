from trainer import Trainer

c = {
    'model_name': ['Resnet18'],'n_epoch': list(range(1,50)),
    'seed': [0], 'bs': 64, 'lr': [1e-4]
}

trainer = Trainer(c)
trainer.run()