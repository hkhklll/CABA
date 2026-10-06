import os
from datetime import datetime
import yaml
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
import logging

os.environ["TQDM_DISABLE"] = "1"
# Set global log level to only show WARNING and above
pl.seed_everything(42, workers=True)
pl_logger = logging.getLogger("pytorch_lightning")
pl_logger.setLevel(logging.WARNING)
# Disable DataLoader progress bar output
pl.utilities.rank_zero_info = lambda *args, **kwargs: None  # Disable PL info level logs


def get_save_name(cfg):
    percentage = cfg['percentage']

    attack_method = 'Normal' if percentage == 0 else cfg['attack']
    if attack_method == 'BadCM' and cfg['badcm'] is not None:
        attack_method = attack_method + cfg['badcm']

    save_name = '{}_{}_{}_{}_p={}_target={}'.format(cfg['module_name'], cfg['backbones'][0],cfg['dataset'], attack_method, percentage, cfg['target'][0])
    return save_name


def run_cmr(module, cfg):

    percentage = cfg['percentage']
    save_name = cfg['save_name']

    print("save_name: {}".format(save_name))

    checkpoint_dir = 'checkpoints/' + save_name
    poison_name = cfg.get('poison_data_last_name', '')
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M')
    ckpt_filename = f'{poison_name}_{timestamp_str}_' + '{epoch:02d}-{step:06d}-{val_map:.4f}' if poison_name else f'{timestamp_str}_' + '{epoch:02d}-{step:06d}-{val_map:.4f}'
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor='val_map',
        dirpath=checkpoint_dir,
        filename=ckpt_filename,
        save_last=False,
        mode='max')

    tb_logger = TensorBoardLogger('log/tensorboard', save_name) if cfg["enable_tb"] else False
    trainer = pl.Trainer(
        devices=len(cfg['device']),
        accelerator='gpu',
        max_epochs=cfg['epochs'],
        check_val_every_n_epoch=cfg["valid_interval"],
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        # Core: disable all progress bar related display
        enable_progress_bar=False,          # Disable PL progress bar
        enable_model_summary=False,        # Disable model summary
        log_every_n_steps=1000,            # Greatly reduce training log frequency
        enable_checkpointing=True,         # Keep checkpoint functionality (no impact)
        # Additional: disable DataLoader worker logs
        num_sanity_val_steps=0            # Disable validation sanity check (reduce startup logs)
    )

    train_loader = module.poi_train_loader if percentage > 0 else module.train_loader
    test_loader = module.test_loader
    """ 
    In PyTorch Lightning, trainer.fit() and trainer.test() automatically call validation_epoch_end and test_epoch_end,
    because this is the framework's preset "hook method" mechanism — through fixed-name methods, it standardizes the execution logic of training, validation, and testing processes, allowing developers to focus on core metric computation without manually writing flow control code.
    """
    if cfg['phase'] == 'train':
        module.flogger.log("=> Training on poisoned data with p={} and target={}".format(percentage, cfg['target']))
        trainer.fit(
            model=module,
            ckpt_path=cfg["checkpoint"],
            train_dataloaders=train_loader,
            val_dataloaders=test_loader
        )

        best_ckpt_path = checkpoint_callback.best_model_path
        if best_ckpt_path:
            yaml_path = best_ckpt_path.replace('.ckpt', '.yaml')
            with open(yaml_path, 'w') as f:
                yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
            module.flogger.log("=> Config saved to: {}".format(yaml_path))

    ckpt = (cfg["checkpoint"] or os.path.join(checkpoint_dir, 'last.ckpt')) if cfg['phase'] == 'test' else 'best'

    if percentage > 0:
        module.flogger.log("=> Testing on poisoned data with p={} and target={}".format(percentage, cfg['target']))
        trainer.test(model=module, dataloaders=module.poi_test_loader, ckpt_path=ckpt)

    module.flogger.log("=> Testing on clean data ...")
    trainer.test(model=module, dataloaders=test_loader, ckpt_path=ckpt)