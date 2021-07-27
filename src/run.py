from trainer import Trainer

c = {
    'model_name': 'test',
    'seed': 0, 'bs': 64, 'lr': 1e-3, 'n_epoch': 1
}

trainer = Trainer(c)
trainer.run()