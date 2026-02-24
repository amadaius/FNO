import argparse
from pathlib import Path
from typing import List, Union, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neuralop.data.datasets.pt_dataset import PTDataset
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.utils import get_project_root


class CFD1DDataProcessor(DefaultDataProcessor):
    def preprocess(self, data_dict, batched=True):
        x = data_dict["x"].to(self.device)
        y = data_dict["y"].to(self.device)

        if self.in_normalizer is not None:
            x_phys = self.in_normalizer.transform(x[:, :3, :])
            x = x.clone()
            x[:, :3, :] = x_phys

        if self.out_normalizer is not None and self.training:
            y = self.out_normalizer.transform(y)

        data_dict["x"] = x
        data_dict["y"] = y
        return data_dict


class CFD1DCompressibleDataset(PTDataset):
    def __init__(
        self,
        root_dir: Union[Path, str],
        n_train: int,
        n_tests: List[int],
        batch_size: int,
        test_batch_sizes: List[int],
        train_resolution: int = 128,
        test_resolutions: List[int] = (128,),
        encode_input: bool = True,
        encode_output: bool = True,
        num_workers: int = 0,
    ):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        for res in list(test_resolutions) + [train_resolution]:
            if res != 128:
                raise ValueError("Only resolution 128 is supported for this dataset wrapper.")

        super().__init__(
            root_dir=root_dir,
            dataset_name="cfd1d",
            n_train=n_train,
            n_tests=n_tests,
            batch_size=batch_size,
            test_batch_sizes=test_batch_sizes,
            train_resolution=train_resolution,
            test_resolutions=list(test_resolutions),
            encode_input=False,
            encode_output=False,
            encoding="channel-wise",
            input_subsampling_rate=None,
            output_subsampling_rate=None,
            channel_dim=1,
            channels_squeezed=False,
        )

        x_train = self.train_db.x
        y_train = self.train_db.y

        in_normalizer = None
        if encode_input:
            in_normalizer = UnitGaussianNormalizer(dim=[0, 2])
            in_normalizer.fit(x_train[:, :3, :])

        out_normalizer = None
        if encode_output:
            out_normalizer = UnitGaussianNormalizer(dim=[0, 2])
            out_normalizer.fit(y_train)

        self._data_processor = CFD1DDataProcessor(
            in_normalizer=in_normalizer, out_normalizer=out_normalizer
        )

        self.num_workers = num_workers


example_data_root = get_project_root() / "neuralop/data/datasets/data/CFD-1D"


def load_cfd1d_compressible_pt(
    data_root: Union[Path, str] = example_data_root,
    n_train: int = 1800,
    n_tests: Optional[List[int]] = None,
    batch_size: int = 16,
    test_batch_sizes: Optional[List[int]] = None,
    train_resolution: int = 128,
    test_resolutions: Optional[List[int]] = None,
    encode_input: bool = True,
    encode_output: bool = True,
    num_workers: int = 0,
):
    if n_tests is None:
        n_tests = [200]
    if test_resolutions is None:
        test_resolutions = [128]
    if test_batch_sizes is None:
        test_batch_sizes = [batch_size]

    dataset = CFD1DCompressibleDataset(
        root_dir=data_root,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions,
        encode_input=encode_input,
        encode_output=encode_output,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    test_loaders = {}
    for res, bsz, n_test in zip(test_resolutions, test_batch_sizes, n_tests):
        test_loaders[res] = DataLoader(
            dataset.test_dbs[res],
            batch_size=bsz,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    return train_loader, test_loaders, dataset.data_processor


def _as_1d(a):
    a = np.asarray(a)
    return a.reshape(-1)


def _pick_key(keys, candidates):
    for k in candidates:
        if k in keys:
            return k
    raise KeyError(f"Missing keys. Available keys: {list(keys)}")


def _to_samples_time_space(arr, nt, nx):
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={a.shape}")

    shape = a.shape

    sample_axis = int(np.argmax(shape))
    rem_axes = [i for i in range(3) if i != sample_axis]

    x_axes = [i for i in rem_axes if shape[i] == nx]
    if len(x_axes) == 1:
        x_axis = x_axes[0]
    else:
        x_axis = max(rem_axes, key=lambda i: shape[i])

    time_axis = [i for i in rem_axes if i != x_axis][0]

    if shape[time_axis] != nt and abs(shape[time_axis] - nt) > 1:
        raise ValueError(
            f"Cannot infer time axis: nt={nt} but inferred time dimension={shape[time_axis]} from shape={shape}"
        )

    a = np.transpose(a, (sample_axis, time_axis, x_axis))
    return a


def _interp_1d_last_dim(x, out_len):
    if x.shape[-1] == out_len:
        return x
    x3 = x.reshape(-1, 1, x.shape[-1])
    y3 = F.interpolate(x3, size=out_len, mode="linear", align_corners=True)
    return y3.reshape(*x.shape[:-1], out_len)


def _ensure_time_len(x, out_len):
    if x.shape[1] == out_len:
        return x
    if x.shape[1] > out_len:
        return x[:, :out_len, :]
    x3 = x.permute(0, 2, 1).reshape(-1, 1, x.shape[1])
    y3 = F.interpolate(x3, size=out_len, mode="linear", align_corners=True)
    y = y3.reshape(x.shape[0], x.shape[2], out_len).permute(0, 2, 1)
    return y


def preprocess_h5_to_pt(
    input_h5: Union[Path, str],
    output_dir: Union[Path, str],
    n_train: int = 1800,
    n_test: int = 200,
    spatial_m: int = 128,
    time_nt: int = 21,
    seed: int = 0,
):
    input_h5 = Path(input_h5).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5.as_posix(), "r") as f:
        keys = set(f.keys())
        xk = _pick_key(keys, ["x-coordinate", "x", "x_coord", "xcoord"])
        tk = _pick_key(keys, ["t-coordinate", "t", "t_coord", "tcoord"])
        dk = _pick_key(keys, ["density", "rho"])
        pk = _pick_key(keys, ["pressure", "p"])
        vk = _pick_key(keys, ["Vx", "vx", "u", "vel", "velocity"])

        x_coord = _as_1d(f[xk][...])
        t_coord = _as_1d(f[tk][...])

        nx0 = int(x_coord.shape[0])
        nt0 = int(t_coord.shape[0])

        rho0 = _to_samples_time_space(f[dk][...], nt=nt0, nx=nx0)
        p0 = _to_samples_time_space(f[pk][...], nt=nt0, nx=nx0)
        vx0 = _to_samples_time_space(f[vk][...], nt=nt0, nx=nx0)

    rho = torch.from_numpy(rho0).float()
    p = torch.from_numpy(p0).float()
    vx = torch.from_numpy(vx0).float()

    rho = _ensure_time_len(rho, time_nt)
    p = _ensure_time_len(p, time_nt)
    vx = _ensure_time_len(vx, time_nt)

    rho = _interp_1d_last_dim(rho, spatial_m)
    p = _interp_1d_last_dim(p, spatial_m)
    vx = _interp_1d_last_dim(vx, spatial_m)

    grid_x = torch.linspace(0.0, 1.0, spatial_m, dtype=torch.float32)

    s = int(rho.shape[0])
    need = int(n_train + n_test)
    if s < need:
        raise ValueError(f"Not enough samples: have {s}, need {need}")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(s, generator=g)[:need]

    rho = rho[perm]
    vx = vx[perm]
    p = p[perm]

    x0 = torch.stack([rho[:, 0, :], vx[:, 0, :], p[:, 0, :]], dim=1)
    grid = grid_x.unsqueeze(0).repeat(need, 1).unsqueeze(1)
    x = torch.cat([x0, grid], dim=1).contiguous()

    y = torch.stack([rho[:, 1:, :], vx[:, 1:, :], p[:, 1:, :]], dim=1)
    y = y.reshape(need, 3 * (time_nt - 1), spatial_m).contiguous()

    x_train = x[:n_train]
    y_train = y[:n_train]
    x_test = x[n_train:n_train + n_test]
    y_test = y[n_train:n_train + n_test]

    train_path = output_dir / f"cfd1d_train_{spatial_m}.pt"
    test_path = output_dir / f"cfd1d_test_{spatial_m}.pt"

    torch.save({"x": x_train, "y": y_train}, train_path.as_posix())
    torch.save({"x": x_test, "y": y_test}, test_path.as_posix())

    print(train_path.as_posix())
    print(test_path.as_posix())
    print(tuple(x_train.shape), tuple(y_train.shape), tuple(x_test.shape), tuple(y_test.shape))


def _main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_h5",
        type=str,
        default=str(Path.home() / "/home/leeshu/wmm/neuraloperator-main/neuralop/data/datasets/data/CFD-1D/Shock/133156"),
        help="Path to raw HDF5 file (e.g. .../CFD-1D/Shock/133156.h5). 默认使用上面的路径，可在命令行覆盖",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(get_project_root() / "neuralop/data/datasets/data/CFD-1D/Shock"),
        help="Directory to write cfd1d_train_128.pt and cfd1d_test_128.pt. 默认使用项目内的 CFD-1D/Shock 目录，可在命令行覆盖",
    )
    p.add_argument("--n_train", type=int, default=1800)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--spatial_m", type=int, default=128)
    p.add_argument("--time_nt", type=int, default=21)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    preprocess_h5_to_pt(
        input_h5=args.input_h5,
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        spatial_m=args.spatial_m,
        time_nt=args.time_nt,
        seed=args.seed,
    )


if __name__ == "__main__":
    _main()