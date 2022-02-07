import torch
import argparse

from exp.exp_rnn import ExpRNN


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameter to program TrajRep")

    parser.add_argument("--name", type=str, required=True, help="datasets name")

    parser.add_argument("--t", type=str, required=True, default="long", help="the length of datasets")

    parser.add_argument("--gpu", type=int, default=0, help="gpu index")
    parser.add_argument("--dis_type", type=str, default="dtw", help="the distance type of dis_matrix", choices=["dtw", "frechet", "hausdorff"])

    parser.add_argument("--origin_in", type=int, default=2, help="origin data dimension")
    parser.add_argument("--d_model", type=int, default=32, help="dimension of model")  # 256

    ###
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=3, help="num of encoder layers")
    parser.add_argument("--d_ff", type=int, default=512, help="dimension of fcn")  # 1024
    parser.add_argument("--dropout", type=float, default=0.005, help="dropout")

    parser.add_argument("--activation", type=str, default="gelu", help="activation name")

    ###

    parser.add_argument("--train_batch_size", type=int, default=20, help="the batch_size of train")
    parser.add_argument("--eval_batch_size", type=int, default=100, help="the batch_size of evaluation (validation or test)")
    parser.add_argument("--matrix_cal_batch", type=int, default=100, help="the number of rows calculating distance matrix")
    parser.add_argument("--num_workers", type=int, default=15, help="the n_work parameter in dataloader")
    parser.add_argument("--sample_num", type=int, default=20, help="the num of samples for each traj")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="optimizer learning rate")
    parser.add_argument("--alpha", type=int, default=16, help="the hyperparameter used in loss computation")
    parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer used")
    parser.add_argument("--epochs", type=int, default=100, help="the number of training epochs")

    parser.add_argument(
        "--data_features", type=float, default=[(108.95058706928398, 0.023035205688164626), (34.24350534114136, 0.01980738220554856)], help="((lon_mean, lon_std), (lat_mean, lat_std)) of the dataset",
    )

    parser.add_argument("--traj_path", type=str, default="", help="the path of trajs")
    parser.add_argument("--matrix_path", type=str, default="", help="the path of dis_matrix")

    parser.add_argument("--train_range", type=int, default=(0, 6000), help="the arange of trajs used in training")
    parser.add_argument("--val_range", type=int, default=(6000, 10000), help="the arange of trajs used in validation")

    parser.add_argument("--model_best_wts_path", type=str, default="./models/wts/{}/{}/{}_{}_wts_hr10 {:.4f}.pkl")

    args = parser.parse_args()

    print("Args in experiment:")
    print(args)

    print(f"\nTraining RNN")
    ExpRNN(args=args).train()

    torch.cuda.empty_cache()


# python -u main.py --name DiDi --t mix --gpu  --dis_type dtw --traj_path '/home/huhaonan/nfs/huhaonan/Data/DiDi/mix/traj/mix_trajs_10000.pkl' --matrix_path '/home/huhaonan/nfs/huhaonan/Data/DiDi/mix/distance/DTW/mix_DiDi_dtw_distance_10000_10000.pkl' # Mix DTW

# python -u main.py --name DiDi --t mix --gpu  --dis_type hausdorff --traj_path '/home/huhaonan/nfs/huhaonan/Data/DiDi/mix/traj/mix_trajs_10000.pkl' --matrix_path '/home/huhaonan/nfs/huhaonan/Data/DiDi/mix/distance/Hausdorff/mix_DiDi_hausdorff_distance_10000_10000.pkl' # Mix Hausdorff


# python -u main.py --name DiDi --t long --gpu  --dis_type dtw --traj_path '/home/huhaonan/nfs/huhaonan/Data/DiDi/long/traj/long_trajs_10000.pkl' --matrix_path '/home/huhaonan/nfs/huhaonan/Data/DiDi/long/distance/DTW/long_DiDi_dtw_distance_10000_10000.pkl' # long DTW


# python -u main.py --name DiDi --t mix --gpu  --dis_type hausdorff --traj_path '/home/huhaonan/nfs/huhaonan/Data/DiDi/long/traj/long_trajs_10000.pkl' --matrix_path '/home/huhaonan/nfs/huhaonan/Data/DiDi/long/distance/Hausdorff/long_DiDi_hausdorff_distance_10000_10000.pkl' # long Hausdorff

