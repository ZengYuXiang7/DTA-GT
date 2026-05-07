import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # ======================== Basic ========================
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--dataset", type=str, default="nasbench201", help="nasbench101||nasbench201")
    parser.add_argument("--predict_target", type=str, default="latency", choices=["accuracy", "latency"],
                        help="nasbench数据集的预测目标: accuracy(val_acc) 或 latency")
    parser.add_argument("--percent", type=float, default=469, help="training samples, percent or numbers")
    
    parser.add_argument("--model", type=str, default="model56")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--do_train", type=bool, default=True)

    # ======================== Data ========================
    parser.add_argument("--data_path", type=str, default="data/nasbench101/all_nasbench101.pt")
    parser.add_argument("--finetuning", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--parallel", action="store_true")

    # ======================== Network: Encoding ========================
    parser.add_argument("--act_function", type=str, default="relu")
    parser.add_argument("--class_token", type=bool, default=True)
    parser.add_argument("--depth_embed", type=bool, default=True)
    parser.add_argument("--encoder_type", type=str, default="nn", help="nn|transformer")
    parser.add_argument("--enc_dim", type=int, default=96, help="Operations encoding dim")
    parser.add_argument("--in_chans", type=int, default=32)

    # ======================== Model (主要调参区) ========================
    parser.add_argument("--d_model", type=int, default=180)
    parser.add_argument("--gcn_layers", type=int, default=10)
    parser.add_argument("--graph_readout", type=str, default="att")
    parser.add_argument("--embed_type", type=str, default="onehot_op", help="architecture token encoding")
    parser.add_argument("--graph_n_head", type=int, default=6)
    
    
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pool_gnn_layers", type=int, default=2)
    parser.add_argument("--num_pooling", type=int, default=1)
    parser.add_argument("--pool_ratio", type=float, default=0.25)

    # ======================== Network: Other ========================
    parser.add_argument("--graph_d_model", type=int, default=160)
    parser.add_argument("--graph_d_ff", type=int, default=640)
    parser.add_argument("--depths", nargs="+", type=int, default=[12])
    parser.add_argument("--drop_path_rate", type=float, default=0.0)
    parser.add_argument("--tf_layers", type=int, default=3)
    parser.add_argument("--use_head", type=int, default=0)
    parser.add_argument("--avg_tokens", action="store_true", help="average tokens for prediction")
    parser.add_argument("--use_aux_loss", type=int, default=0)

    # ======================== Optimizer ========================
    group = parser.add_argument_group("Optimizer parameters")
    group.add_argument("--opt", default="adamw", type=str, help='Optimizer (default: "adamw")')
    group.add_argument("--opt-eps", default=None, type=float, help="Optimizer Epsilon")
    group.add_argument("--opt-betas", default=None, type=float, nargs="+", help="Optimizer Betas")
    group.add_argument("--momentum", type=float, default=0.9, help="Optimizer momentum")
    group.add_argument("--weight_decay", type=float, default=0.01)

    # ======================== Learning Rate Schedule ========================
    group = parser.add_argument_group("Learning rate schedule parameters")
    group.add_argument("--sched", default="cosine", type=str, help='LR scheduler (default: "cosine")')
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--lr_cycle_mul", type=float, default=1.0)
    group.add_argument("--min_ratio", type=float, default=1e-1, help="lower lr bound for cyclic schedulers")
    group.add_argument("--decay_rate", "--dr", type=float, default=0.1, help="LR decay rate")
    group.add_argument("--warmup-lr", type=float, default=1e-6)
    group.add_argument("--warmup-epochs", type=int, default=5)
    group.add_argument("--warmup_step", type=float, default=0.1)
    group.add_argument("--lr-cycle-limit", type=int, default=1)
    group.add_argument("--epochs", type=int, default=3000)

    # ======================== Training ========================
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--save_path", type=str, default="model/")
    parser.add_argument("--save_epoch_freq", type=int, default=1000)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--print_freq", type=int, default=200)
    parser.add_argument("--patience", type=int, default=1500, help="early stopping patience, <=0 disables")

    # ======================== Loss ========================
    parser.add_argument("--lambda_mse", type=float, default=1.0, help="weight of mse loss")
    parser.add_argument("--lambda_rank", type=float, default=0.8, help="weight of ranking loss")
    parser.add_argument("--lambda_consistency", type=float, default=0.0, help="weight of consistency loss")
    
    # ======================== Evaluation ========================
    parser.add_argument("--test_freq", type=int, default=1)
    parser.add_argument("--tqdm", default=0, type=int, help="use tqdm for training progress bar")
    parser.add_argument("--try_exp", default=-1, type=int, help="try experiment id")

    args, unknown_args = parser.parse_known_args()
    return args
