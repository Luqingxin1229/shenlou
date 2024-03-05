from Deep_SAD import DeepSAD


device = 'cuda:3'


deepSAD.train(dataset,
              optimizer_name=cfg.settings['optimizer_name'],
              lr=cfg.settings['lr'],
              n_epochs=cfg.settings['n_epochs'],
              lr_milestones=cfg.settings['lr_milestone'],
              batch_size=cfg.settings['batch_size'],
              weight_decay=cfg.settings['weight_decay'],
              device=device,
              n_jobs_dataloader=n_jobs_dataloader)