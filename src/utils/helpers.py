import wandb
import pytorch_lightning as pl
from matplotlib import pyplot as plt


class SizeDatamodule(pl.callbacks.Callback):
    """Log the size of the dataset at the beginning of the training."""

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """At each epoch start, log the size of the train set."""
        if pl_module.logger:
            pl_module.logger.log_metrics(
                {
                    "sanity_check/train-size": len(trainer.train_dataloader.dataset),
                    "sanity_check/val-size": len(trainer.val_dataloaders.dataset),
                },
                step=0,
            )

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """At each test start, log the size of the test set."""
        if pl_module.logger:
            pl_module.logger.log_metrics(
                {"sanity_check/test-size": len(trainer.test_dataloaders.dataset)},
                step=0,
            )


class SanityCheckInput(pl.callbacks.Callback):
    """Plots the input data at the beginning of the training."""

    def on_sanity_check_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """At each epoch start, log the size of the train set."""
        iteration = iter(trainer.val_dataloaders)
        point_cloud, features, _ = next(iteration)
        show(point_cloud, features, pl_module.logger, machine=pl_module.cfg.paths.name)

    # def on_validation_epoch_end(
    #     self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    # ):
    #     iteration = iter(trainer.val_dataloaders)
    #     inputs, _ = next(iteration)
    #     show(inputs, pl_module.logger, machine=pl_module.cfg.paths.___config_name___,
    # name=pl_module.global_step)


def show(inputs, features, logger, machine) -> None:
    """Show point cloud."""
    views = len(inputs) if isinstance(inputs, list) else 1

    # batch = inputs[0].shape[0] if views > 1 else inputs.shape[0]
    fig, axs = plt.subplots(ncols=views * 2, nrows=5, squeeze=True, figsize=(15, 15))
    # choices = np.random.choice(len(ground_truth), 4)
    # - for 5 in the batch
    for idx in range(5):
        nb = idx
        for idx_col in range(views):
            pc = inputs[idx_col][nb] if views > 1 else inputs[nb]
            feature = features[idx_col][nb] if views > 1 else features[nb]
            axs[idx, idx_col].scatter(pc[:, 0], pc[:, 1], s=1, c="rosybrown", alpha=0.5)
            axs[idx, idx_col + 1].imshow(
                feature.cpu().numpy().reshape(50, 50), cmap="plasma"
            )

    # - log sanity checks
    if isinstance(logger, pl.loggers.WandbLogger):
        logger.experiment.log({"sanity_check/input_data": wandb.Image(fig)})
    elif machine == "didion":
        fig.savefig("checks/input_data.png")
