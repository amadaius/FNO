import argparse
import ast

import torch
from torch.utils.data import DataLoader

from neuralop.models import FNO
from neuralop.data.datasets.tensor_dataset import TensorDataset
from neuralop import H1Loss


def _parse_n_modes(s: str):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--n_modes 需要形如 '16,16'，但得到: {s}")
    return (int(parts[0]), int(parts[1]))


def rel_l2_allchannels(y_pred, y_true, eps: float):
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    y_true = y_true.reshape(y_true.shape[0], -1)
    diff = y_pred - y_true
    num = torch.linalg.vector_norm(diff, ord=2, dim=-1)
    den = torch.linalg.vector_norm(y_true, ord=2, dim=-1)
    return torch.mean(num / (den + eps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_pt", type=str, default=None)
    p.add_argument("--test_pt", type=str, default=None)
    p.add_argument("--stats_pt", type=str, default=None, help="cfd2d_norm_stats_*.pt，包含 y_mean/y_std（用于物理空间loss）")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--test_batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--hidden_channels", type=int, default=32)
    p.add_argument("--n_modes", type=str, default="16,16", help="形如 '16,16'，需 <= 32（对64分辨率）")

    p.add_argument("--in_channels", type=int, default=6)
    p.add_argument("--out_channels", type=int, default=80)

    p.add_argument("--positional_embedding", type=str, default="none", choices=["none", "grid"])
    p.add_argument("--fno_skip", type=str, default="linear")
    p.add_argument("--use_channel_mlp", type=str, default="True")

    p.add_argument("--hc_rate", type=int, default=2)
    p.add_argument("--hc_dynamic", type=str, default="True")

    p.add_argument("--rel_eps", type=float, default=1e-8)
    p.add_argument("--verbose", dest="verbose", type=str, default="True")

    p.add_argument("--wandb.log", dest="wandb_log", type=str, default="False")
    p.add_argument("--wandb.project", dest="wandb_project", type=str, default=None)
    p.add_argument("--wandb.name", dest="wandb_name", type=str, default=None)

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

    verbose = args.verbose.lower() in ("true", "1", "yes")
    wandb_log = args.wandb_log.lower() in ("true", "1", "yes")
    hc_dynamic = args.hc_dynamic.lower() in ("true", "1", "yes")
    use_channel_mlp = args.use_channel_mlp.lower() in ("true", "1", "yes")

    if args.train_pt is None or args.test_pt is None:
        raise ValueError("Provide --train_pt/--test_pt")
    if args.stats_pt is None:
        raise ValueError("Provide --stats_pt (cfd2d_norm_stats_*.pt)")

    n_modes = _parse_n_modes(args.n_modes)
    if n_modes[0] > 32 or n_modes[1] > 32:
        raise ValueError(f"64分辨率下建议 n_modes <= 32，但得到 n_modes={n_modes}")

    train_data = torch.load(args.train_pt)
    test_data = torch.load(args.test_pt)
    stats = torch.load(args.stats_pt)

    x_train = train_data["x"].float()
    y_train = train_data["y"].float()
    x_test = test_data["x"].float()
    y_test = test_data["y"].float()

    if x_train.ndim != 4 or y_train.ndim != 4:
        raise ValueError(f"期望 x/y 为4D张量 [B,C,H,W]，但得到 x={tuple(x_train.shape)} y={tuple(y_train.shape)}")

    train_resolution = int(x_train.shape[-1])

    if x_train.shape[1] != args.in_channels:
        raise ValueError(f"in_channels不匹配: 期望 {args.in_channels}，但 x.shape[1]={x_train.shape[1]}")
    if y_train.shape[1] != args.out_channels:
        raise ValueError(f"out_channels不匹配: 期望 {args.out_channels}，但 y.shape[1]={y_train.shape[1]}")

    y_mean = stats["y_mean"].float()
    y_std = stats["y_std"].float()

    if y_mean.ndim != 1 or y_std.ndim != 1 or y_mean.shape[0] != args.out_channels or y_std.shape[0] != args.out_channels:
        raise ValueError(
            f"stats_pt 里 y_mean/y_std 维度不符合 out_channels={args.out_channels}："
            f" y_mean={tuple(y_mean.shape)} y_std={tuple(y_std.shape)}"
        )

    if verbose:
        print("##### CONFIG #####\n")
        print(
            {
                "n_layers": args.n_layers,
                "n_modes": n_modes,
                "hidden_channels": args.hidden_channels,
                "batch_size": args.batch_size,
                "test_batch_size": args.test_batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "in_channels": args.in_channels,
                "out_channels": args.out_channels,
                "positional_embedding": args.positional_embedding,
                "hc_rate": args.hc_rate,
                "hc_dynamic": hc_dynamic,
            }
        )
        print(f"train: x {tuple(x_train.shape)} y {tuple(y_train.shape)}")
        print(f"test : x {tuple(x_test.shape)} y {tuple(y_test.shape)}")

    train_db = TensorDataset(x=x_train, y=y_train)
    test_db = TensorDataset(x=x_test, y=y_test)

    train_loader = DataLoader(train_db, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_db, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)

    positional_embedding = None if args.positional_embedding == "none" else "grid"

    model = FNO(
        n_modes=n_modes,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        positional_embedding=positional_embedding,
        use_channel_mlp=use_channel_mlp,
        fno_skip=args.fno_skip,
        hc_rate=args.hc_rate,
        hc_dynamic=hc_dynamic,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    h1loss = H1Loss(d=2)

    y_mean_d = y_mean.view(1, -1, 1, 1).to(device)
    y_std_d = y_std.view(1, -1, 1, 1).to(device)

    def to_phys(y_norm):
        return y_norm * y_std_d + y_mean_d

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            pred = model(x)

            pred_phys = to_phys(pred)
            y_phys = to_phys(y)

            loss = rel_l2_allchannels(pred_phys, y_phys, eps=args.rel_eps)
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
                x = batch["x"].to(device)
                y = batch["y"].to(device)

                pred = model(x)

                pred_phys = to_phys(pred)
                y_phys = to_phys(y)

                err_l2 = rel_l2_allchannels(pred_phys, y_phys, eps=args.rel_eps)
                err_h1 = h1loss(pred_phys, y_phys)

                test_rel_l2 += float(err_l2.detach().cpu())
                test_h1 += float(err_h1.detach().cpu())
                n_test_batches += 1

        train_loss /= max(1, n_train_batches)
        test_rel_l2 /= max(1, n_test_batches)
        test_h1 /= max(1, n_test_batches)

        print(
            f"Eval(direct): {train_resolution}_h1={test_h1:.4f}, {train_resolution}_l2={test_rel_l2:.4f}"
        )

        if wandb_log:
            import wandb

            if epoch == 1:
                wandb.init(project=args.wandb_project or "neuralop", name=args.wandb_name or "cfd2d_mhc_fno")
            wandb.log(
                {"epoch": epoch, "train_rel_l2_phys": train_loss, "test_rel_l2_phys": test_rel_l2, "test_h1": test_h1},
                commit=False,
            )


if __name__ == "__main__":
    main()