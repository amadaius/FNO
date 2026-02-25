import argparse

import torch
from torch.utils.data import DataLoader

from neuralop.models import FNO
from neuralop.data.datasets.tensor_dataset import TensorDataset
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop import H1Loss


def rel_l2(y_pred, y_true, eps: float):
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    y_true = y_true.reshape(y_true.shape[0], -1)
    diff = y_pred - y_true
    num = torch.linalg.vector_norm(diff, ord=2, dim=-1)
    den = torch.linalg.vector_norm(y_true, ord=2, dim=-1)
    return torch.mean(num / (den + eps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_pt", type=str, required=True)
    p.add_argument("--test_pt", type=str, required=True)

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--n_modes", type=int, default=16)

    p.add_argument("--use_channel_mlp", type=str, default="True")
    p.add_argument("--hc_rate", type=int, default=2)
    p.add_argument("--hc_dynamic", type=str, default="True")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rel_eps", type=float, default=1e-8)
    p.add_argument("--normalize_x", type=str, default="True")
    p.add_argument("--verbose", type=str, default="True")
    args, _ = p.parse_known_args()

    torch.manual_seed(args.seed)
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")

    verbose = args.verbose.lower() in ("true", "1", "yes")
    use_channel_mlp = args.use_channel_mlp.lower() in ("true", "1", "yes")
    hc_dynamic = args.hc_dynamic.lower() in ("true", "1", "yes")
    normalize_x = args.normalize_x.lower() in ("true", "1", "yes")

    train_data = torch.load(args.train_pt)
    test_data = torch.load(args.test_pt)

    x_train = train_data["x"].float()
    y_train = train_data["y"].float()
    x_test = test_data["x"].float()
    y_test = test_data["y"].float()

    assert x_train.shape == (1000, 256, 2)
    assert y_train.shape == (1000, 256, 1)
    assert x_test.shape == (200, 256, 2)
    assert y_test.shape == (200, 256, 1)

    x_train = x_train.permute(0, 2, 1).contiguous()  # (N, 2, 256)
    y_train = y_train.permute(0, 2, 1).contiguous()  # (N, 1, 256)
    x_test = x_test.permute(0, 2, 1).contiguous()
    y_test = y_test.permute(0, 2, 1).contiguous()

    train_db = TensorDataset(x_train, y_train)
    test_db = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_db, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_db, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)

    x_norm = None
    if normalize_x:
        x_norm = UnitGaussianNormalizer(dim=[0, 2])
        x_norm.fit(x_train)
        x_norm = x_norm.to(device)

    def preprocess_x(x):
        x = x.to(device)
        if x_norm is not None:
            x = x_norm.transform(x)
        return x

    model = FNO(
        n_modes=(args.n_modes,),
        in_channels=2,
        out_channels=1,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        positional_embedding=None,
        use_channel_mlp=use_channel_mlp,
        fno_skip="linear",
        lifting_channel_ratio=2.0,
        projection_channel_ratio=2.0,
        hc_rate=args.hc_rate,
        hc_dynamic=hc_dynamic,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    h1loss = H1Loss(d=1)

    if verbose:
        print("##### CONFIG #####")
        print(
            {
                "train_pt": args.train_pt,
                "test_pt": args.test_pt,
                "n_layers": args.n_layers,
                "n_modes": args.n_modes,
                "hidden_channels": args.hidden_channels,
                "batch_size": args.batch_size,
                "test_batch_size": args.test_batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "hc_rate": args.hc_rate,
                "hc_dynamic": hc_dynamic,
                "use_channel_mlp": use_channel_mlp,
                "normalize_x": normalize_x,
                "device": str(device),
            }
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            x = preprocess_x(batch["x"])
            y = batch["y"].to(device)
            pred = model(x)
            loss = rel_l2(pred, y, eps=args.rel_eps)
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
                x = preprocess_x(batch["x"])
                y = batch["y"].to(device)
                pred = model(x)
                err_l2 = rel_l2(pred, y, eps=args.rel_eps)
                err_h1 = h1loss(pred, y)
                test_rel_l2 += float(err_l2.detach().cpu())
                test_h1 += float(err_h1.detach().cpu())
                n_test_batches += 1

        train_loss /= max(1, n_train_batches)
        test_rel_l2 /= max(1, n_test_batches)
        test_h1 /= max(1, n_test_batches)

        print(f"[Epoch {epoch:04d}] train_rel_l2={train_loss:.6f} test_rel_l2={test_rel_l2:.6f} test_h1={test_h1:.6f}")


if __name__ == "__main__":
    main()