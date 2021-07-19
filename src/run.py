from trainer import Trainer

c = {
    'model_name': 'test',
    'seed': 0, 'bs': 8, 'lr': 1e-3, 'n_epoch': 10
}

trainer = Trainer(c)
trainer.run()