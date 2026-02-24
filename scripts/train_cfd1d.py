import argparse
import ast

import torch
from torch.utils.data import DataLoader

from neuralop.models import FNO
from neuralop.data.datasets.tensor_dataset import TensorDataset
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop import H1Loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_pt", type=str, default=None)
    p.add_argument("--test_pt", type=str, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--n_modes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--positional_embedding", type=str, default="none", choices=["none", "grid"])
    p.add_argument("--hc_rate", type=int, default=0)
    p.add_argument("--hc_dynamic", type=str, default="True")
    p.add_argument("--fno_skip", type=str, default="linear")
    p.add_argument("--use_channel_mlp", type=str, default="True")
    p.add_argument("--prediction_strategy", type=str, default="direct", choices=["direct", "autoregressive"])
    p.add_argument("--ar_train_mode", type=str, default="teacher_forcing", choices=["teacher_forcing", "rollout"])
    p.add_argument("--rel_eps", type=float, default=1e-8)
    p.add_argument("--lifting_channel_ratio", type=float, default=2.0)
    p.add_argument("--projection_channel_ratio", type=float, default=2.0)
    p.add_argument("--data.folder", dest="data_folder", type=str, default=None)
    p.add_argument("--data.train_resolution", dest="data_train_resolution", type=int, default=128)
    p.add_argument("--data.test_resolutions", dest="data_test_resolutions", type=str, default="[128]")
    p.add_argument("--data.n_train", dest="data_n_train", type=int, default=1800)
    p.add_argument("--data.n_tests", dest="data_n_tests", type=str, default="[200]")
    p.add_argument("--data.batch_size", dest="data_batch_size", type=int, default=None)
    p.add_argument("--data.test_batch_sizes", dest="data_test_batch_sizes", type=str, default=None)
    p.add_argument("--data.encode_input", dest="data_encode_input", type=str, default="True")
    p.add_argument("--data.encode_output", dest="data_encode_output", type=str, default="True")
    p.add_argument("--data.download", dest="data_download", type=str, default="False")
    p.add_argument("--wandb.log", dest="wandb_log", type=str, default="False")
    p.add_argument("--wandb.project", dest="wandb_project", type=str, default=None)
    p.add_argument("--wandb.name", dest="wandb_name", type=str, default=None)
    p.add_argument("--verbose", dest="verbose", type=str, default="True")
    p.add_argument("--opt.learning_rate", dest="opt_learning_rate", type=float, default=None)
    p.add_argument("--opt.weight_decay", dest="opt_weight_decay", type=float, default=None)
    p.add_argument("--opt.n_epochs", dest="opt_n_epochs", type=int, default=None)
    args, _ = p.parse_known_args()

    torch.manual_seed(args.seed)
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")

    if args.opt_learning_rate is not None:
        args.lr = args.opt_learning_rate
    if args.opt_weight_decay is not None:
        args.weight_decay = args.opt_weight_decay
    if args.opt_n_epochs is not None:
        args.epochs = args.opt_n_epochs
    if args.data_batch_size is not None:
        args.batch_size = args.data_batch_size
    if args.data_test_batch_sizes:
        try:
            tb = ast.literal_eval(args.data_test_batch_sizes)
            if isinstance(tb, (list, tuple)) and len(tb) > 0:
                args.test_batch_size = int(tb[0])
        except Exception:
            pass

    encode_input = args.data_encode_input.lower() in ("true", "1", "yes")
    encode_output = args.data_encode_output.lower() in ("true", "1", "yes")
    wandb_log = args.wandb_log.lower() in ("true", "1", "yes")
    verbose = args.verbose.lower() in ("true", "1", "yes")
    if verbose:
        print("##### CONFIG #####\n")
        print({
            "n_layers": args.n_layers,
            "n_modes": args.n_modes,
            "hidden_channels": args.hidden_channels,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "train_resolution": args.data_train_resolution,
        })

    if (args.train_pt is None or args.test_pt is None) and args.data_folder:
        res = args.data_train_resolution
        args.train_pt = f"{args.data_folder}/cfd1d_train_{res}.pt"
        args.test_pt = f"{args.data_folder}/cfd1d_test_{res}.pt"
    if args.train_pt is None or args.test_pt is None:
        raise ValueError("Provide --train_pt/--test_pt or --data.folder")

    train_data = torch.load(args.train_pt)
    test_data = torch.load(args.test_pt)

    x_train = train_data["x"].float()
    y_train = train_data["y"].float()
    x_test = test_data["x"].float()
    y_test = test_data["y"].float()

    train_db = TensorDataset(x_train, y_train)
    test_db = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_db, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_db, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)

    def rel_l2_allchannels(y_pred, y_true, eps: float):
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        y_true = y_true.reshape(y_true.shape[0], -1)
        diff = y_pred - y_true
        num = torch.linalg.vector_norm(diff, ord=2, dim=-1)
        den = torch.linalg.vector_norm(y_true, ord=2, dim=-1)
        return torch.mean(num / (den + eps))

    in_norm = UnitGaussianNormalizer(dim=[0, 2]) if encode_input else None
    if in_norm is not None:
        in_norm.fit(x_train[:, :3, :])
        in_norm = in_norm.to(device)

    if y_train.shape[1] % 3 != 0:
        raise ValueError(f"Expected y channels divisible by 3 (rho,vx,p), got y.shape={tuple(y_train.shape)}")
    n_future_steps = y_train.shape[1] // 3

    prediction_strategy = args.prediction_strategy
    if prediction_strategy == "autoregressive":
        out_channels = 3
        out_norm = UnitGaussianNormalizer(dim=[0, 2, 3]) if encode_output else None
        if out_norm is not None:
            out_norm.fit(y_train.view(y_train.shape[0], 3, n_future_steps, y_train.shape[-1]))
            out_norm = out_norm.to(device)
    else:
        out_channels = y_train.shape[1]
        out_norm = UnitGaussianNormalizer(dim=[0, 2]) if encode_output else None
        if out_norm is not None:
            out_norm.fit(y_train)
            out_norm = out_norm.to(device)

    positional_embedding = None if args.positional_embedding == "none" else "grid"
    hc_dynamic = args.hc_dynamic.lower() in ("true", "1", "yes")
    use_channel_mlp = args.use_channel_mlp.lower() in ("true", "1", "yes")

    model = FNO(
        n_modes=(args.n_modes,),
        in_channels=4,
        out_channels=out_channels,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        positional_embedding=positional_embedding,
        use_channel_mlp=use_channel_mlp,
        fno_skip=args.fno_skip,
        lifting_channel_ratio=args.lifting_channel_ratio,
        projection_channel_ratio=args.projection_channel_ratio,
        hc_rate=args.hc_rate,
        hc_dynamic=hc_dynamic,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    h1loss = H1Loss(d=1)

    def preprocess_x(x):
        x = x.to(device)
        if in_norm is not None:
            x_phys = in_norm.transform(x[:, :3, :])
            x = x.clone()
            x[:, :3, :] = x_phys
        return x

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            opt.zero_grad(set_to_none=True)

            if prediction_strategy == "autoregressive":
                x0 = batch["x"].to(device)
                grid = x0[:, 3:4, :]
                y = batch["y"].to(device)
                y_steps = y.view(y.shape[0], 3, n_future_steps, y.shape[-1])

                prev_state = x0[:, :3, :]
                loss_acc = 0.0
                for t in range(n_future_steps):
                    x_in = torch.cat([prev_state, grid], dim=1)
                    x_in = preprocess_x(x_in)

                    pred = model(x_in)
                    if out_norm is not None:
                        pred = out_norm.inverse_transform(pred)

                    target = y_steps[:, :, t, :]
                    loss_acc = loss_acc + rel_l2_allchannels(pred, target, eps=args.rel_eps)

                    if args.ar_train_mode == "rollout":
                        prev_state = pred
                    else:
                        prev_state = target

                loss = loss_acc / float(n_future_steps)
            else:
                x = preprocess_x(batch["x"])
                y = batch["y"].to(device)

                pred = model(x)
                if out_norm is not None:
                    pred = out_norm.inverse_transform(pred)

                loss = rel_l2_allchannels(pred, y, eps=args.rel_eps)

            loss.backward()
            opt.step()

            train_loss += float(loss.detach().cpu())
            n_train_batches += 1

        model.eval()
        test_rel_l2 = 0.0
        test_h1 = 0.0
        n_test_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                if prediction_strategy == "autoregressive":
                    x0 = batch["x"].to(device)
                    grid = x0[:, 3:4, :]
                    y = batch["y"].to(device)
                    y_steps = y.view(y.shape[0], 3, n_future_steps, y.shape[-1])

                    prev_state = x0[:, :3, :]
                    preds = []
                    h1_acc = 0.0
                    for t in range(n_future_steps):
                        x_in = torch.cat([prev_state, grid], dim=1)
                        x_in = preprocess_x(x_in)

                        pred = model(x_in)
                        if out_norm is not None:
                            pred = out_norm.inverse_transform(pred)

                        target = y_steps[:, :, t, :]
                        h1_acc += float(h1loss(pred, target).detach().cpu())
                        preds.append(pred)
                        prev_state = pred

                    pred_traj = torch.stack(preds, dim=2)
                    err_l2 = rel_l2_allchannels(pred_traj, y_steps, eps=args.rel_eps)
                    err_h1 = h1_acc / float(n_future_steps)
                else:
                    x = preprocess_x(batch["x"])
                    y = batch["y"].to(device)

                    pred = model(x)
                    if out_norm is not None:
                        pred = out_norm.inverse_transform(pred)

                    err_l2 = rel_l2_allchannels(pred, y, eps=args.rel_eps)
                    err_h1 = h1loss(pred, y)

                test_rel_l2 += float(err_l2.detach().cpu())
                test_h1 += float(err_h1 if isinstance(err_h1, float) else err_h1.detach().cpu())
                n_test_batches += 1

        train_loss /= max(1, n_train_batches)
        test_rel_l2 /= max(1, n_test_batches)
        test_h1 /= max(1, n_test_batches)

        print(f"Eval({prediction_strategy}): {args.data_train_resolution}_h1={test_h1:.4f}, {args.data_train_resolution}_l2={test_rel_l2:.4f}")
        if wandb_log:
            import wandb
            if epoch == 1:
                wandb.init(project=args.wandb_project or "neuralop", name=args.wandb_name or "cfd1d_run")
            wandb.log({"epoch": epoch, "train_rel_l2_phys": train_loss, "test_rel_l2_phys": test_rel_l2, "test_h1": test_h1}, commit=False)


if __name__ == "__main__":
    main()