from train import experiment
exp = experiment(batchSize=2, cuda=True, load_from_ckpt=True)
exp.train()
