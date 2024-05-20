# sin filtro
def main():
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)

    import pandas as pd
    import torch
    from pytorch_lightning.loggers import WandbLogger
    from helper_functions import count_classes

    from models.image_classifier_module import ImageClassificationLightningModule
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from data.data_modules import CovidDataModule, Sampling
    from torchmetrics.classification import (
        MulticlassAccuracy,
        MulticlassConfusionMatrix,
        MulticlassPrecision,
        MulticlassRecall,
    )
    from torchmetrics import MetricCollection
    from models.mlp import MLP, get_mlp_transformations

    from torch import nn
    import wandb
    import configuration as config

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_count = count_classes(config.ROOT_DIR)

    metrics = MetricCollection(
        {
            "Accuracy": MulticlassAccuracy(num_classes=class_count, average="micro"),
            "BalancedAccuracy": MulticlassAccuracy(num_classes=class_count),
            "Precision": MulticlassPrecision(num_classes=class_count),
            "Recall": MulticlassRecall(num_classes=class_count),
        }
    )
    vector_metrics = MetricCollection(
        {
            "Accuracy": MulticlassAccuracy(num_classes=class_count, average=None),
            "Precision": MulticlassPrecision(num_classes=class_count, average=None),
            "Recall": MulticlassRecall(num_classes=class_count, average=None),
            "Confusion Matrix": MulticlassConfusionMatrix(num_classes=class_count),
        }
    )

    train_transform, test_transform = get_mlp_transformations()

    covid_dm = CovidDataModule(
        root_dir=config.MLP_FEATURES_DIR,
        batch_size=config.BATCH_SIZE,
        test_size=config.TEST_SIZE,
        use_index=config.USE_INDEX,
        indices_dir=config.INDICES_DIR,
        sampling=Sampling.NONE,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    covid_dm.prepare_data()
    covid_dm.create_data_loaders()

    metrics_data = []
    for i in range(config.NUM_TRIALS):
        mlp = MLP(
            input_size=3072,
            hidden_layer_count=10,
            hidden_layer_size=256,
            output_size=class_count,
            device=device,
        )

        model = ImageClassificationLightningModule(
            model=mlp,
            loss_fn=nn.CrossEntropyLoss(),
            metrics=metrics,
            vectorized_metrics=vector_metrics,
            lr=config.LR,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
        )

        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=config.PATIENCE,
            strict=False,
            verbose=False,
            mode="min",
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            dirpath=config.MLP_DIR,
            filename=config.MLP_FILENAME + str(i),
            save_top_k=config.TOP_K_SAVES,
            mode="min",
        )

        id = config.MLP_FILENAME + str(i) + "_" + wandb.util.generate_id()
        wandb_logger = WandbLogger(project=config.WANDB_PROJECT, id=id, resume="allow")

        trainer = Trainer(
            logger=wandb_logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            max_epochs=config.EPOCHS,
            log_every_n_steps=1,
        )

        trainer.fit(model, datamodule=covid_dm)

        # save the metrics per class as well as the confusion matrix to a csv file
        metrics_data.append(trainer.test(model, datamodule=covid_dm)[0])

        results_per_class_metrics = model.test_vect_metrics_result

        metrics_per_class = pd.DataFrame(
            {
                "Accuracy": results_per_class_metrics["Accuracy"].cpu().numpy(),
                "Precision": results_per_class_metrics["Precision"].cpu().numpy(),
                "Recall": results_per_class_metrics["Recall"].cpu().numpy(),
            },
            index=config.CLASS_NAMES,
        )

        confusion_matrix = pd.DataFrame(
            results_per_class_metrics["Confusion Matrix"].cpu().numpy(),
            index=config.CLASS_NAMES,
            columns=config.CLASS_NAMES,
        )

        metrics_per_class.to_csv(config.MLP_CSV_PER_CLASS_FILENAME)
        confusion_matrix.to_csv(config.MLP_CSV_CM_FILENAME)
        wandb.finish()

    pd.DataFrame(metrics_data).to_csv(config.MLP_CSV_FILENAME, index=False)


if __name__ == "__main__":
    main()
