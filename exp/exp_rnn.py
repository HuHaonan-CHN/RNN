import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter


import copy
import time
import datetime

from models.rnn import RNN
from exp.exp_basic import ExpBasic
from utils.data_loader import get_data_loader
from models.loss import WeightedRankingLoss
from models.accuracy_functions import get_embedding_acc

from utils.tools import pload, pdump


class ExpRNN(ExpBasic):
    def __init__(self, args):
        super(ExpRNN, self).__init__(args)
        self.log_writer = SummaryWriter(f"./runs/{self.args.name}/{self.args.t}/{self.args.t}_{self.args.dis_type}_{datetime.date.today()}/")

    def _build_model(self):
        model = RNN(input_size=self.args.origin_in, hidden_size=self.args.d_model, device=self.device).float()

        return model

    def _get_data(self, flag):
        if flag == "train":
            trajs = pload(self.args.traj_path)[self.args.train_range[0] : self.args.train_range[1]]
            matrix = pload(self.args.matrix_path)[self.args.train_range[0] : self.args.train_range[1], self.args.train_range[0] : self.args.train_range[1]]
        elif flag == "val":
            trajs = pload(self.args.traj_path)
            matrix = pload(self.args.matrix_path)[self.args.val_range[0] : self.args.val_range[1], :]
        elif flag == "test":
            trajs = pload(self.args.test_traj_path)
            matrix = pload(self.args.test_dis_matrix_path.format(self.args.dis_type))
        elif flag == "embed":
            nn = 5000
            trajs = pload(self.args.val_traj_path)[:nn]
            matrix = pload(self.args.val_dis_matrix_path.format(self.args.dis_type))[:nn, :nn]
            flag = "val"

        data_loader = get_data_loader(trajs, matrix, flag, self.args.data_features, self.args.sample_num, self.args.train_batch_size, self.args.eval_batch_size, self.args.num_workers)

        return data_loader

    def _select_optimizer(self, optimizer):
        if optimizer == "Adam":
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif optimizer == "SGD":
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)

        return model_optim

    def _select_criterion(self):
        criterion = WeightedRankingLoss(self.args.sample_num, self.args.alpha, self.device).float()

        return criterion

    def embed(self):
        embedding_time = time.time()
        embed_loader = self._get_data(flag="embed")
        all_vectors = []

        self.model.eval()

        for trajs, _, traj_lengths, _ in embed_loader:
            trajs = trajs.to(self.device)

            with torch.no_grad():
                vectors = self.model(trajs, traj_lengths)  # vecters [B,N_S,D]

            all_vectors.append(vectors.squeeze(1))

        all_vectors = torch.cat(all_vectors, dim=0)

        print("Embeddings shape:", all_vectors.shape)
        embedding_time = time.time() - embedding_time
        print(f"!!!!Embedding Time: {embedding_time} seconds.")
        return all_vectors

    def val(self):
        val_loader = self._get_data(flag="val")
        all_vectors = []

        self.model.eval()

        for trajs, traj_lengths, _ in val_loader:
            trajs = trajs.to(self.device)

            with torch.no_grad():
                vectors = self.model(trajs, traj_lengths)  # vecters [B,N_S,D]

            all_vectors.append(vectors.squeeze(1))

        all_vectors = torch.cat(all_vectors, dim=0)

        hr10, hr50, r10_50 = get_embedding_acc(row_embedding_tensor=all_vectors[self.args.val_range[0] : self.args.val_range[1]], col_embedding_tensor=all_vectors, distance_matrix=val_loader.dataset.dis_matrix, matrix_cal_batch=self.args.matrix_cal_batch,)

        return hr10, hr50, r10_50

    def train(self):
        train_loader = self._get_data(flag="train")

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_hr10 = 0.0
        # best_embeddings = None
        time_now = time.time()

        model_optim = self._select_optimizer(self.args.optimizer)
        criterion = self._select_criterion()

        for epoch in range(self.args.epochs):
            self.model.train()

            epoch_begin_time = time.time()
            epoch_loss = 0.0

            for trajs, traj_lengths, dis in train_loader:
                model_optim.zero_grad()

                trajs = trajs.to(self.device)

                with torch.set_grad_enabled(True):
                    vectors = self.model(trajs, traj_lengths)  # vecters [B,N_S,D]

                loss = criterion(vectors, torch.tensor(dis).to(self.device))

                loss.backward()
                model_optim.step()

                epoch_loss += loss

            epoch_loss = epoch_loss / len(train_loader.dataset)
            self.log_writer.add_scalar(f"RNN/Loss", float(epoch_loss), epoch)

            epoch_end_time = time.time()
            print(f"\nEpoch {epoch+1}/{self.args.epochs}:\nTrain Loss: {epoch_loss:.4f}\tTime: {(epoch_end_time - epoch_begin_time) // 60} m {int((epoch_end_time - epoch_begin_time) % 60)} s")

            val_begin_time = time.time()
            hr10, hr50, r10_50 = self.val()
            val_end_time = time.time()

            # this_embeddings = self.embed()

            self.log_writer.add_scalar(f"RNN/HR10", hr10, epoch)
            self.log_writer.add_scalar(f"RNN/HR50", hr50, epoch)
            self.log_writer.add_scalar(f"RNN/R10@50", r10_50, epoch)

            print(f"Val HR10: {100 * hr10:.4f}%\tHR50: {100 * hr50:.4f}%\tR10@50: {100 * r10_50:.4f}%\tTime: {(val_end_time -val_begin_time) // 60} m {int((val_end_time -val_begin_time) % 60)} s")

            if hr10 > best_hr10:
                best_hr10 = hr10
                best_model_wts = copy.deepcopy(self.model.state_dict())
                # best_embeddings = this_embeddings.cpu().numpy()

        time_end = time.time()

        # pdump(best_embeddings, f"Neutraj_{self.args.t}_Hausdorff_embeddings_{best_embeddings.shape[0]}_{best_embeddings.shape[1]}.pkl")
        # pdump(best_embeddings, f"Neutraj_long_Hausdorff_embeddings_{best_embeddings.shape[0]}_{best_embeddings.shape[1]}.pkl")

        print("\nAll training complete in {:.0f}m {:.0f}s".format((time_end - time_now) // 60, (time_end - time_now) % 60))
        print(f"Best HR10: {100*best_hr10:.4f}%")

        torch.save(best_model_wts, self.args.model_best_wts_path.format(self.args.name, self.args.t, "RNN", self.args.dis_type, best_hr10))

        self.model.load_state_dict(best_model_wts)

        return self.model

    def test(self, setting):
        pass

    def predict(self, setting, load=False):
        pass

