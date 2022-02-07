from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class MyDataset(Dataset):
    def __init__(self, data, dis_matrix, phase, data_features, sample_num):
        super(MyDataset).__init__()

        (lon_mean, lon_std), (lat_mean, lat_std) = data_features
        self.dis_matrix = dis_matrix / dis_matrix.max()
        self.phase = phase
        self.data = []
        self.sample_num = sample_num // 2 * 2

        for i in range(len(data)):
            traj = torch.tensor(data[i])
            traj = traj - torch.tensor([lon_mean, lat_mean])
            traj = traj * torch.tensor([1 / lon_std, 1 / lat_std])
            self.data.append(traj.float())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.phase == "train":
            sample_index = []
            traj_list = []
            dis_list = []

            id_list = np.argsort(self.dis_matrix[index])
            # 取最相似的几个
            sample_index.extend(id_list[: self.sample_num // 2])
            # 取最不相似的几个
            sample_index.extend(id_list[len(id_list) - self.sample_num // 2 :])

            for i in sample_index:
                traj_list.append(self.data[i])
                dis_list.append(self.dis_matrix[sample_index[0], i])
        elif self.phase == "val" or "test":
            traj_list = [self.data[index]]
            dis_list = None

        return traj_list, dis_list


def _pad_traj(traj, max_length):
    _, D = traj.shape
    padding_traj = torch.zeros((max_length - len(traj), D))
    traj = torch.vstack((traj, padding_traj))
    return traj.numpy().tolist()


def _generate_square_subsequent_mask(traj_num, max_length, lenths):
    """
    padding_mask
    lenths: [lenth1,lenth2...]
    """
    mask = torch.ones(traj_num, max_length) == 1  # mask batch_size*sample_num x max_lenth

    for i, this_lenth in enumerate(lenths):
        for j in range(this_lenth):
            mask[i][j] = False

    return mask


def _get_standard_inputs(inputs):
    """
    inputs 
    [
        [traj, traj, traj]
        [traj, traj, traj]
    ]
    """
    max_length = 0
    traj_tensor = []
    traj_length = []

    for b in inputs:
        for t in b:
            traj_length.append(len(t))
            if len(t) > max_length:
                max_length = len(t)

    for b_trajs in inputs:
        temp_b_trajs = []
        for traj in b_trajs:
            temp_b_trajs.append(_pad_traj(traj, max_length))
        traj_tensor.append(temp_b_trajs)

    traj_tensor = torch.tensor(traj_tensor, dtype=torch.float)  # traj_tensor [N, sample_num, S, 2]

    padding_mask = _generate_square_subsequent_mask(len(traj_length), max_length, traj_length)  # mask batch_size*sample_num x max_lenth

    return traj_tensor, padding_mask, traj_length


def _my_collect_function(batch):
    traj_list_all = []
    dis_list_all = []

    for i in batch:
        traj_list = i[0]
        dis_list = i[1]
        traj_list_all.append(traj_list)
        dis_list_all.append(dis_list)

    traj_list_all, padding_mask, traj_lengths = _get_standard_inputs(traj_list_all)  # inputs [batch_size, sample_num, sequence_len, embedding_size]  padding_mask=[traj_num, sequence_len]

    return traj_list_all, traj_lengths, dis_list_all


def get_data_loader(data, dis_matrix, phase, data_features, sample_num, train_batch_size, eval_batch_size, num_workers):
    dataset = MyDataset(data, dis_matrix, phase, data_features, sample_num)
    if phase == "train":
        is_shuffle = True
        batch_size = train_batch_size
    elif phase == "val" or "test":
        is_shuffle = False
        batch_size = eval_batch_size

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers, collate_fn=_my_collect_function)

    return data_loader

