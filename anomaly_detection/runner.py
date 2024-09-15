import os
import datetime
import torch
import torch.optim as optim
import logging
import numpy as np
import hydra
from omegaconf import DictConfig
from aeon.classification.interval_based import TimeSeriesForestClassifier

from models import (
    attentionMTSC, CNNLSTM_lw, CNNLSTM_dn, simpleCNN, CNN_cw, VisionModel
)
from tools.utils import train, test, is_supervised
from tools.dataset import get_dataloader
from tools.plot import (
    plot_loss, plot_accuracy, plot_confusion_matrix, vizualize_imputation_batch
)


sansserif = {'fontname': 'sans-serif'}


def load_model(data_cfg, model_cfg, task):
    series_len = int(data_cfg.preproc.window_size * data_cfg.sample_frequency)

    if model_cfg.name == "attention":
        return attentionMTSC(
            series_len=series_len,
            input_dim=data_cfg.num_features,
            learnable_pos_enc=model_cfg.learnable_pos_enc,
            classes=data_cfg.num_classes,
            task=task,
            d_model=model_cfg.d_model,
            heads=model_cfg.heads,
            dropout=model_cfg.dropout,
            dim_ff=model_cfg.feed_forward_dim,
            num_layers=model_cfg.num_layers,
            weights_fp=model_cfg.weights_fp
        )
    elif model_cfg.name == "cnn_lw":
        return CNNLSTM_lw(
            data_cfg.num_features,
            model_cfg.num_filters,
            model_cfg.hidden_size,
            data_cfg.num_classes
        )
    elif model_cfg.name == "cnn_dn":
        return CNNLSTM_dn(
            data_cfg.num_features,
            series_len,
            model_cfg.num_filters,
            model_cfg.hidden_size,
            data_cfg.num_classes
        )
    elif model_cfg.name == "cnn":
        return simpleCNN(
            data_cfg.num_features,
            series_len,
            model_cfg.num_filters,
            data_cfg.num_classes
        )
    elif model_cfg.name == "tsf":
        return TimeSeriesForestClassifier(n_estimators=model_cfg.num_trees)
    elif model_cfg.name == "cnn_cw":
        return CNN_cw(
            input_dim=data_cfg.num_features,
            classes=data_cfg.num_classes,
            time_steps=series_len,
            num_filters1=model_cfg.nfilters1,
            num_filters2=model_cfg.nfilters2,
            pool1=model_cfg.pool1,
            pool2=model_cfg.pool2,
            dim_ff=model_cfg.dimff
        )
    elif model_cfg.name == "vision":
        return VisionModel(
            backbone=model_cfg.backbone,
            hidden_dim=model_cfg.hidden_dim,
            dim_ff=model_cfg.dim_ff,
            series_len=series_len,
            classes=data_cfg.num_classes)
    else:
        raise Exception("unrecognized model.")


def is_early(mod_name):
    early = ['tsf']
    return mod_name in early


class Runner:
    def __init__(self, data_cfg, model_cfg, task, lr, num_epochs, weight_decay,
                 output_path, log=True, plots=False):
        self.dataset_name = data_cfg.name
        self.model_name = model_cfg.name
        self.task = task
        self.num_epochs = num_epochs
        self.output_path = output_path
        self.logging = log
        self.plots = plots
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(data_cfg, model_cfg, task)
        if not is_early(self.model_name):
            self.model.to(self.device)
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_torch(self, train_loader, val_loader):
        train_losses = []
        val_losses = []
        train_acc = []
        val_acc = []
        best_acc = 0

        for epoch in range(self.num_epochs):
            out = train(
                epoch=epoch,
                num_epochs=self.num_epochs,
                model=self.model,
                optimizer=self.optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                task=self.task
            )

            train_mean_loss, train_mean_acc = out[0], out[2]
            val_mean_loss, val_mean_acc = out[1], out[3]

            # loss summarizing
            train_losses.append(train_mean_loss)
            val_losses.append(val_mean_loss)
            train_acc.append(train_mean_acc)
            val_acc.append(val_mean_acc)

            # --------------------------
            # Logging Stage
            # --------------------------
            if self.logging:
                logging.info(
                    f"Epoch {epoch + 1}: average loss - {val_mean_loss:.6f}")

            print("Epoch: ", epoch + 1)
            print("train_loss: {}, train_class_acc: {}"
                  .format(train_mean_loss, train_mean_acc))
            print("val_loss: {}, val_class_acc: {}"
                  .format(val_mean_loss, val_mean_acc))

            if val_mean_acc >= best_acc:
                best_acc = val_mean_acc
                self.save_model()

                print("Best model is saved.")

            if (epoch + 1) % 10 == 0 and self.plots and \
                    not is_supervised(self.task):
                epoch_results = out[-1]

                save_folder = os.path.join(
                    "./plots", self.dataset_name, "imputation",
                    self.model_name
                )
                os.makedirs(save_folder, exist_ok=True)
                vizualize_imputation_batch(
                    preds=epoch_results['preds'],
                    targets=epoch_results['targets'],
                    masks=epoch_results['masks'],
                    out_fp=os.path.join(
                        save_folder, f"imputation_{epoch + 1}.png")
                )

        if self.plots:
            plots_dir = os.path.join(
                "./plots", self.dataset_name, "process_curve",
                self.model_name, self.task
            )
            os.makedirs(plots_dir, exist_ok=True)
            plot_loss(train_losses, val_losses,
                      os.path.join(plots_dir, "loss_graph.png"))
            plot_accuracy(train_acc, val_acc,
                          os.path.join(plots_dir, "acc_graph.png"))

    def ds2np(self, ds):
        X, Y = [], []
        for _, (x, y) in enumerate(ds):
            X.extend(x.numpy())
            Y.extend(y.numpy())

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def train_early(self, ds):
        trainX, trainY = self.ds2np(ds)
        self.model.fit(trainX, trainY)

    def test_early(self, ds):
        testX, testY = self.ds2np(ds)
        pred_classes = self.model.predict(testX)
        classes, counts = np.unique(pred_classes == testY, return_counts=True)
        acc = {c: count / np.sum(counts) for c, count in zip(classes, counts)}

        if self.plots and is_supervised(self.task):
            save_folder = os.path.join(
                "./plots", self.dataset_name, "confusion_matrix",
                self.model_name, self.task
            )
            os.makedirs(save_folder, exist_ok=True)
            plot_confusion_matrix(
                pred_classes, testY, os.path.join(save_folder, "cm.png"))

        final_acc = acc[True]
        if self.logging:
            logging.info(f"FINAL MODEL ACCURACY: {final_acc*100:.2f}%")
        print(f"FINAL MODEL ACCURACY: {final_acc*100:.2f}%")

    def test_torch(self, test_loader):
        weights_fp = os.path.join(
                self.output_path, self.dataset_name,
                f"{self.model_name}_model_weights_{self.task}.pth"
            )

        weight = torch.load(weights_fp)
        self.model.load_state_dict(weight)
        print("Loaded best model weight from", weights_fp)

        mean_acc, epoch_results = test(
            self.model, test_loader, self.device, self.task, plots=self.plots
        )

        if self.plots and is_supervised(self.task):
            y_preds = epoch_results['preds']
            y_trues = epoch_results['targets']
            save_folder = os.path.join(
                "./plots", self.dataset_name, "confusion_matrix",
                self.model_name, self.task
            )
            os.makedirs(save_folder, exist_ok=True)
            plot_confusion_matrix(
                y_preds, y_trues, os.path.join(save_folder, "cm.png"))

        if self.logging:
            logging.info(f"FINAL MODEL ACCURACY: {mean_acc*100:.2f}%")
        print(f"FINAL MODEL ACCURACY: {mean_acc*100:.2f}%")

    def train_model(self, *dataloader):
        if self.logging:
            logging.info("======== TRAINING MODEL ======== ")
        if is_early(self.model_name):
            self.train_early(dataloader[0])
        else:
            self.train_torch(*dataloader)

    def test_model(self, dataloader):
        if self.logging:
            logging.info("======== TESTING MODEL ======== ")
        if is_early(self.model_name):
            self.test_early(dataloader)
        else:
            self.test_torch(dataloader)

    def get_model(self):
        return self.model

    def save_torch(self, output_path):
        torch.save(self.model.state_dict(), output_path)

    def save_early(self, output_path):
        self.model.save(output_path)

    def save_model(self):
        if self.logging:
            logging.info("======== SAVING MODEL ======== ")
        os.makedirs(os.path.join(
            self.output_path, self.dataset_name), exist_ok=True)
        if is_early(self.model_name):
            output_fp = os.path.join(
                self.output_path, self.dataset_name,
                f"{self.model_name}_model_weights_{self.task}"
            )
            self.save_early(output_fp)
        else:
            output_fp = os.path.join(
                self.output_path, self.dataset_name,
                f"{self.model_name}_model_weights_{self.task}.pth"
            )
            self.save_torch(output_fp)


@hydra.main(config_path="conf", config_name="", version_base="1.3")
def run(cfg: DictConfig):
    if cfg.task == "imputation":
        assert cfg.model.name == "attention", \
            "Imputation task only supports attention model."

    if cfg.model.name == "vision":
        assert cfg.task == "classification", "Task must be classification."
        assert cfg.dataset.name == "PoolData", "Dataset must be PoolData."

    if cfg.log:
        def modLog(modName: str, params: dict) -> None:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            if logger.hasHandlers():
                logger.handlers.clear()
            dir_fp = "./logs"
            os.makedirs(dir_fp, exist_ok=True)
            log_fp = os.path.join(dir_fp, f"{modName}.log")
            file_handler = logging.FileHandler(log_fp)
            logger.addHandler(file_handler)
            current_time = datetime.datetime.now()
            current_time.strftime("%D : %I:%M:%S %p")
            logging.info("===========================================")
            logging.info(f"###          MODEL: {modName}        ###")
            logging.info(f"###    START_TIME: {current_time}    ###")
            logging.info(" loading data. ")
            logging.info("###      Parameters.    ###")
            for key, value in params.items():
                logging.info(f"--> {key} : {value}")
        modLog(cfg.model.name, cfg)
        logging.info("======== MODEL CONFIGURATION ======== ")
        logging.info(f"task: {cfg.task}.")
        logging.info(f"device: {cfg.device}.")
        logging.info(f"======== LOADING DATASET: {cfg.dataset.name} ======== ")

    train_loader, val_loader, test_loader = get_dataloader(
        cfg.dataset, cfg.hyp.batch_size, cfg.device.num_workers,
        cfg.model.name, task=cfg.task, plot=cfg.plots)

    runner = Runner(
        data_cfg=cfg.dataset,
        model_cfg=cfg.model,
        task=cfg.task,
        lr=cfg.hyp.learning_rate,
        num_epochs=cfg.hyp.num_epochs,
        weight_decay=cfg.hyp.weight_decay,
        output_path=cfg.path.output_path,
        log=cfg.log,
        plots=cfg.plots)
    runner.train_model(train_loader, val_loader)
    runner.test_model(test_loader)


if __name__ == "__main__":
    run()
