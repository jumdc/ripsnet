import pytorch_lightning as pl


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
