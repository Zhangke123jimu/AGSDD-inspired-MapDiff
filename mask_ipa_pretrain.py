from comet_ml import Experiment
import hydra
import os
import torch

from dataloader.large_dataset import Cath
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from dataloader.collator import CollatorIPAPretrain
from model.ipa.ipa_net import IPANetPredictor
from torch.optim import Adam, lr_scheduler
from trainer.mask_ipa_trainer import Trainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_ddp():
    distributed = "LOCAL_RANK" in os.environ

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    return distributed,local_rank, rank, world_size,rank==0

@hydra.main(version_base=None, config_path="conf", config_name="mask_pretrain")
def main(cfg: DictConfig):
    distributed, local_rank, rank, world_size,is_main_process = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if cfg.comet.use and is_main_process:
        experiment = Experiment(
            project_name=cfg.comet.project_name,
            workspace=cfg.comet.workspace,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False
        )
        experiment.log_parameters(OmegaConf.to_container(cfg))
        experiment.set_name(cfg.experiment.name)
        if cfg.comet.comet_tag:
            experiment.add_tag(cfg.comet.comet_tag)
    else:
        experiment = None
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if is_main_process:
        print(OmegaConf.to_yaml(cfg))
        print(f"Output directory: {output_dir}")
    if experiment and is_main_process:
        experiment.log_parameters({"output_dir": output_dir})

    train_ID = os.listdir(cfg.dataset.train_dir)

    train_dataset = Cath(train_ID, cfg.dataset.train_dir)

    train_sampler = DistributedSampler(train_dataset,shuffle=True) if distributed else None

    collator = CollatorIPAPretrain(candi_rate=cfg.train.candi_rate, mask_rate=cfg.train.mask_rate,
                                   replace_rate=cfg.train.replace_rate, keep_rate=cfg.train.keep_rate)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=(train_sampler is None), num_workers=12, sampler=train_sampler,
                              collate_fn=collator)

    model = IPANetPredictor(dropout=cfg.model.ipa_drop_out).to(device)
    resume_checkpoint=None
    if cfg.resume_path is not None:
        try:
            resume_checkpoint = torch.load(cfg.resume_path,map_location=device)
            model.load_state_dict(resume_checkpoint['model'], strict=False)
            if is_main_process:
                print(f"Resuming from checkpoint {cfg.resume_path}: \n"
                      f"  Global epoch: {resume_checkpoint['global_epoch']} \n"
                      f"  Current train loss: {resume_checkpoint['loss']} \n" )

        except Exception as e:
            raise RuntimeError(f"Failed to resume from {cfg.resume_path}") from e

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = Adam(model.parameters(), lr=cfg.train.lr, betas=(0.95, 0.999), weight_decay=cfg.train.weight_decay)
    if cfg.resume_path is not None:
        optimizer.load_state_dict(resume_checkpoint['opt'])

    steps_per_epoch = len(train_loader)
    if cfg.train.scheduler:
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.train.lr, total_steps=cfg.train.train_epochs * steps_per_epoch)
    else:
        scheduler = None

    if cfg.resume_path is not None and scheduler is not None and resume_checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(resume_checkpoint["scheduler"])

    if is_main_process:
        print(f"Total parameters: {count_parameters(model)}")

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    trainer = Trainer(config=cfg, model=model, optimizer=optimizer, epochs=cfg.train.train_epochs, loss_fn=loss_fn,
                      train_dataloader=train_loader, output_dir=output_dir, device=device,
                      scheduler=scheduler, experiment=experiment, resume_path=cfg.resume_path,distributed=distributed,
                      is_main_process=is_main_process,train_sampler=train_sampler)
    trainer.fit()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    if is_main_process:
        print(f"Output directory: {output_dir}")
        os.system("bash auto_save_shutdown.sh")

if __name__ == "__main__":
    main()
