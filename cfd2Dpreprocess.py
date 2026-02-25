import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AxisSpec:
    n_axis: int
    t_axis: int
    h_axis: int
    w_axis: int
    n_total: int
    t_total: int
    h: int
    w: int


def _collect_dataset_paths(h5: h5py.File) -> List[str]:
    paths: List[str] = []

    def _visitor(name: str, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    h5.visititems(_visitor)
    return sorted(paths)


def _basename(path: str) -> str:
    return path.split("/")[-1] if "/" in path else path


def _pick_dataset_path(
    all_paths: Sequence[str],
    preferred: Optional[str],
    candidates: Sequence[str],
) -> str:
    if preferred is not None and preferred.strip():
        if preferred in all_paths:
            return preferred
        raise KeyError(
            f"手动指定的数据集路径不存在: {preferred}\n"
            f"可用数据集示例(前20个): {list(all_paths)[:20]}"
        )

    cand_l = [c.lower() for c in candidates]

    exact_matches: List[str] = []
    fuzzy_matches: List[str] = []

    for p in all_paths:
        b = _basename(p).lower()
        if b in cand_l:
            exact_matches.append(p)
        else:
            for c in cand_l:
                if c in b:
                    fuzzy_matches.append(p)
                    break

    if exact_matches:
        exact_matches = sorted(exact_matches, key=lambda s: (len(s), s))
        return exact_matches[0]

    if fuzzy_matches:
        fuzzy_matches = sorted(fuzzy_matches, key=lambda s: (len(s), s))
        return fuzzy_matches[0]

    raise KeyError(
        f"无法自动找到数据集。候选关键字: {list(candidates)}\n"
        f"可用数据集示例(前50个): {list(all_paths)[:50]}"
    )


def _infer_axes_from_shape(
    shape: Tuple[int, ...],
    t_expected: int,
    spatial_sizes: Optional[Tuple[int, int]] = None,
    n_expected: Optional[int] = None,
) -> AxisSpec:
    if len(shape) != 4:
        raise ValueError(f"期望变量数据集为 4D 的某种排列，但得到 shape={shape}")

    dims = list(shape)

    t_hits = [i for i, d in enumerate(dims) if d == t_expected]
    if len(t_hits) != 1:
        raise ValueError(f"无法唯一定位时间轴 T={t_expected}，shape={shape}, hits={t_hits}")
    t_axis = t_hits[0]

    remaining = [i for i in range(4) if i != t_axis]
    rem_sizes = [dims[i] for i in remaining]

    spatial_axes: Optional[Tuple[int, int]] = None

    if spatial_sizes is not None:
        hs, ws = int(spatial_sizes[0]), int(spatial_sizes[1])
        need = [hs, ws]
        hits: List[int] = []
        for s in need:
            cand = [i for i in remaining if dims[i] == s and i not in hits]
            if cand:
                hits.append(cand[0])
        if len(hits) == 2:
            spatial_axes = (hits[0], hits[1])
        elif hs == ws:
            cand = [i for i in remaining if dims[i] == hs]
            if len(cand) >= 2:
                spatial_axes = (cand[0], cand[1])

    if spatial_axes is None:
        # 常见情形：两个空间维相等(如 64x64 / 512x512)
        size_to_axes: Dict[int, List[int]] = {}
        for i in remaining:
            size_to_axes.setdefault(dims[i], []).append(i)
        equal_pairs = [axes for axes in size_to_axes.values() if len(axes) >= 2]
        if equal_pairs:
            axes = sorted(equal_pairs, key=lambda a: (-len(a), dims[a[0]]))[-1]
            spatial_axes = (axes[0], axes[1])

    if spatial_axes is None:
        # 兜底：取剩余维度中最小的两个当空间轴（N 往往更大）
        sorted_axes = sorted(remaining, key=lambda i: dims[i])
        spatial_axes = (sorted_axes[0], sorted_axes[1])

    ha, wa = spatial_axes
    n_candidates = [i for i in remaining if i not in spatial_axes]
    if len(n_candidates) != 1:
        raise ValueError(f"无法唯一确定 N 轴，shape={shape}, t_axis={t_axis}, spatial_axes={spatial_axes}")
    n_axis = n_candidates[0]

    if n_expected is not None and dims[n_axis] < int(n_expected):
        raise ValueError(
            f"样本数不足: N_total={dims[n_axis]} < n_expected={int(n_expected)}，shape={shape}。"
            "请调整 --n-total/--n-train/--n-test，或提供包含更多样本的 HDF5 文件。"
        )

    return AxisSpec(
        n_axis=n_axis,
        t_axis=t_axis,
        h_axis=ha,
        w_axis=wa,
        n_total=dims[n_axis],
        t_total=dims[t_axis],
        h=dims[ha],
        w=dims[wa],
    )


def _quick_infer_n_total(shape: Tuple[int, ...], t_expected: int) -> Optional[int]:
    if len(shape) != 4:
        return None

    dims = list(shape)
    t_hits = [i for i, d in enumerate(dims) if d == t_expected]
    if len(t_hits) != 1:
        return None

    t_axis = t_hits[0]
    rem = [dims[i] for i in range(4) if i != t_axis]

    dup = None
    for v in set(rem):
        if rem.count(v) >= 2:
            dup = v
            break

    if dup is not None:
        others = [v for v in rem if v != dup]
        if len(others) == 1:
            return int(others[0])

    return int(max(rem))


def _read_var_time_slice_THW(
    ds: h5py.Dataset,
    axis: AxisSpec,
    i: int,
    t_sel,
    t_len_expected: Optional[int],
) -> np.ndarray:
    index = [slice(None)] * 4
    index[axis.n_axis] = i
    index[axis.t_axis] = t_sel
    arr = np.asarray(ds[tuple(index)], dtype=np.float32)

    if arr.ndim == 2:
        return arr  # [H,W]（空间维次序不重要）

    if arr.ndim != 3:
        raise ValueError(f"期望读到 3D [T,H,W] 的某种排列，但得到 shape={arr.shape}")

    if t_len_expected is None:
        t_axis_local = 0
    else:
        hits = [k for k, d in enumerate(arr.shape) if d == t_len_expected]
        if len(hits) != 1:
            raise ValueError(
                f"无法唯一定位局部时间轴(长度={t_len_expected})，arr.shape={arr.shape}, hits={hits}"
            )
        t_axis_local = hits[0]

    if t_axis_local != 0:
        perm = [t_axis_local] + [k for k in range(3) if k != t_axis_local]
        arr = np.transpose(arr, perm)

    return arr  # [T,H,W]


def _make_grid_hw2(h: int, w: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    xi = torch.linspace(0.0, 1.0, h, dtype=dtype)
    eta = torch.linspace(0.0, 1.0, w, dtype=dtype)
    xx, yy = torch.meshgrid(xi, eta, indexing="ij")
    return torch.stack([xx, yy], dim=-1)  # [H,W,2]


def _downsample_hwC_area(x_hwC: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if x_hwC.ndim != 3:
        raise ValueError(f"期望 [H,W,C]，但得到 {tuple(x_hwC.shape)}")
    h, w, c = x_hwC.shape
    if (h, w) == (target_h, target_w):
        return x_hwC
    x = x_hwC.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
    x = F.interpolate(x, size=(target_h, target_w), mode="area")
    return x.squeeze(0).permute(1, 2, 0).contiguous()  # [target_h,target_w,C]


def _compute_channel_stats_from_stream(
    ds_map: Dict[str, h5py.Dataset],
    axis: AxisSpec,
    n_samples: int,
    mode: str,
    eps: float,
    target_res: int,
    log_every: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == "x":
        c = 4
        t_sel = 0
        t_len_expected = None
    elif mode == "y":
        c = 80
        t_sel = slice(1, 21)  # t=1..20
        t_len_expected = 20
    else:
        raise ValueError(f"未知 mode={mode}，只支持 'x' 或 'y'")

    sum_c = np.zeros((c,), dtype=np.float64)
    sumsq_c = np.zeros((c,), dtype=np.float64)
    count = 0

    keys = ["density", "pressure", "vx", "vy"]

    for i in range(n_samples):
        if mode == "x":
            vars_hw4 = []
            for k in keys:
                hw = _read_var_time_slice_THW(ds_map[k], axis, i, t_sel, t_len_expected)
                vars_hw4.append(hw)
            x_hw4 = torch.from_numpy(np.stack(vars_hw4, axis=-1))  # [H,W,4]
            x_hw4 = _downsample_hwC_area(x_hw4, target_res, target_res)
            flat = x_hw4.reshape(-1, 4).double().cpu().numpy()
        else:
            vars_thw4 = []
            for k in keys:
                thw = _read_var_time_slice_THW(ds_map[k], axis, i, t_sel, t_len_expected=t_len_expected)  # [20,H,W]
                vars_thw4.append(thw)
            y_thw4 = np.stack(vars_thw4, axis=-1).astype(np.float32)  # [20,H,W,4]
            y_hw80 = np.transpose(y_thw4, (1, 2, 0, 3)).reshape(axis.h, axis.w, 80)  # [H,W,80]
            y_hw80_t = torch.from_numpy(y_hw80)
            y_hw80_t = _downsample_hwC_area(y_hw80_t, target_res, target_res)
            flat = y_hw80_t.reshape(-1, 80).double().cpu().numpy()

        sum_c += flat.sum(axis=0)
        sumsq_c += (flat * flat).sum(axis=0)
        count += flat.shape[0]

        if (i + 1) % log_every == 0 or (i + 1) == n_samples:
            print(f"[统计] mode={mode} 处理进度: {i + 1}/{n_samples}")

    mean = sum_c / float(count)
    ex2 = sumsq_c / float(count)
    var = np.maximum(ex2 - mean * mean, 0.0)
    std = np.sqrt(var) + float(eps)

    mean_t = torch.from_numpy(mean.astype(np.float32))
    std_t = torch.from_numpy(std.astype(np.float32))
    return mean_t, std_t


def _build_and_normalize_tensors(
    ds_map: Dict[str, h5py.Dataset],
    axis: AxisSpec,
    n_total_take: int,
    n_train: int,
    n_test: int,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    target_res: int,
    log_every: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h = target_res
    w = target_res
    grid_hw2 = _make_grid_hw2(h, w, dtype=torch.float32)

    train_x = torch.empty((n_train, 6, h, w), dtype=torch.float32)
    train_y = torch.empty((n_train, 80, h, w), dtype=torch.float32)
    test_x = torch.empty((n_test, 6, h, w), dtype=torch.float32)
    test_y = torch.empty((n_test, 80, h, w), dtype=torch.float32)

    keys = ["density", "pressure", "vx", "vy"]

    for i in range(n_total_take):
        vars_hw4 = []
        for k in keys:
            hw = _read_var_time_slice_THW(ds_map[k], axis, i, 0, t_len_expected=None)  # [Hraw,Wraw]
            vars_hw4.append(hw)
        x_hw4 = torch.from_numpy(np.stack(vars_hw4, axis=-1).astype(np.float32))  # [Hraw,Wraw,4]
        x_hw4 = _downsample_hwC_area(x_hw4, h, w)
        x_hw4 = (x_hw4 - x_mean) / x_std
        x_hw6 = torch.cat([x_hw4, grid_hw2], dim=-1)  # [H,W,6]
        x_chw = x_hw6.permute(2, 0, 1).contiguous()  # [6,H,W]

        vars_thw4 = []
        for k in keys:
            thw = _read_var_time_slice_THW(ds_map[k], axis, i, slice(1, 21), t_len_expected=20)  # [20,Hraw,Wraw]
            vars_thw4.append(thw)
        y_thw4 = np.stack(vars_thw4, axis=-1).astype(np.float32)  # [20,Hraw,Wraw,4]
        y_hw80 = np.transpose(y_thw4, (1, 2, 0, 3)).reshape(axis.h, axis.w, 80)  # [Hraw,Wraw,80]
        y_hw80_t = torch.from_numpy(y_hw80)
        y_hw80_t = _downsample_hwC_area(y_hw80_t, h, w)
        y_hw80_t = (y_hw80_t - y_mean) / y_std
        y_chw = y_hw80_t.permute(2, 0, 1).contiguous()  # [80,H,W]

        if i < n_train:
            train_x[i].copy_(x_chw)
            train_y[i].copy_(y_chw)
        else:
            j = i - n_train
            test_x[j].copy_(x_chw)
            test_y[j].copy_(y_chw)

        if (i + 1) % log_every == 0 or (i + 1) == n_total_take:
            print(f"[构建] 处理进度: {i + 1}/{n_total_take}")

    return train_x, train_y, test_x, test_y


def preprocess_pdebench_cfd2d_to_pt(
    h5_path: str,
    out_dir: str,
    target_res: int = 64,
    n_total: int = 2000,
    n_train: int = 1800,
    n_test: int = 200,
    t_total: int = 21,
    eps: float = 1e-6,
    density_path: Optional[str] = None,
    pressure_path: Optional[str] = None,
    vx_path: Optional[str] = None,
    vy_path: Optional[str] = None,
) -> None:
    if n_train + n_test != n_total:
        raise ValueError(f"划分不一致: n_train+n_test={n_train+n_test} 但 n_total={n_total}")
    if target_res <= 0:
        raise ValueError(f"target_res 必须为正整数，但得到 {target_res}")

    print(f"正在打开 HDF5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        all_paths = _collect_dataset_paths(f)
        if not all_paths:
            raise RuntimeError("HDF5 内未找到任何 Dataset，请检查文件是否损坏或结构不同。")

        print(f"HDF5 内发现 Dataset 数量: {len(all_paths)}")
        print("Dataset 示例(前30个):")
        for p in all_paths[:30]:
            print(f"  - {p}")

        density_p = _pick_dataset_path(all_paths, density_path, ["density", "rho"])
        pressure_p = _pick_dataset_path(all_paths, pressure_path, ["pressure", "p"])
        vx_p = _pick_dataset_path(all_paths, vx_path, ["vx", "u", "velx", "velocity_x"])
        vy_p = _pick_dataset_path(all_paths, vy_path, ["vy", "v", "vely", "velocity_y"])

        print("已选择的数据集路径:")
        print(f"  Density : {density_p}")
        print(f"  Pressure: {pressure_p}")
        print(f"  Vx      : {vx_p}")
        print(f"  Vy      : {vy_p}")

        ds_map = {
            "density": f[density_p],
            "pressure": f[pressure_p],
            "vx": f[vx_p],
            "vy": f[vy_p],
        }

        raw_shape = tuple(ds_map["density"].shape)
        n_in_file = _quick_infer_n_total(raw_shape, t_total)
        if n_in_file is None:
            print(f"该 HDF5 文件样本数 N=未知 (density.shape={raw_shape})")
        else:
            print(f"该 HDF5 文件样本数 N={n_in_file} (density.shape={raw_shape})")

        spatial_sizes = None
        if "x-coordinate" in f and "y-coordinate" in f:
            try:
                x_shape = tuple(f["x-coordinate"].shape)
                y_shape = tuple(f["y-coordinate"].shape)
                x_len = int(x_shape[0] if len(x_shape) == 1 else x_shape[-1])
                y_len = int(y_shape[0] if len(y_shape) == 1 else y_shape[-1])
                if x_len > 0 and y_len > 0:
                    spatial_sizes = (y_len, x_len)
            except Exception:
                spatial_sizes = None

        axis = _infer_axes_from_shape(
            tuple(ds_map["density"].shape),
            t_expected=t_total,
            spatial_sizes=spatial_sizes,
            n_expected=n_total,
        )
        for k, ds in ds_map.items():
            if tuple(ds.shape) != tuple(ds_map["density"].shape):
                raise ValueError(
                    f"变量 {k} 的 shape={ds.shape} 与 density.shape={ds_map['density'].shape} 不一致，无法对齐堆叠。"
                )

        n_total_take = min(axis.n_total, n_total)
        if n_total_take != n_total:
            print(f"注意: 数据集 N_total={axis.n_total}，将仅取前 {n_total_take} 个样本用于导出。")

        print(
            "推断到的轴信息:"
            f" n_axis={axis.n_axis}, t_axis={axis.t_axis}, h_axis={axis.h_axis}, w_axis={axis.w_axis},"
            f" raw_shape={tuple(ds_map['density'].shape)}, raw_hw=({axis.h},{axis.w}), target_res={target_res}"
        )

        print("开始仅用 Train(x) 计算 x_mean / x_std ...")
        x_mean, x_std = _compute_channel_stats_from_stream(
            ds_map, axis, n_train, mode="x", eps=eps, target_res=target_res, log_every=100
        )
        print(f"x_mean shape={tuple(x_mean.shape)}  x_std shape={tuple(x_std.shape)}")

        print("开始仅用 Train(y) 计算 y_mean / y_std ...")
        y_mean, y_std = _compute_channel_stats_from_stream(
            ds_map, axis, n_train, mode="y", eps=eps, target_res=target_res, log_every=50
        )
        print(f"y_mean shape={tuple(y_mean.shape)}  y_std shape={tuple(y_std.shape)}")

        print("开始构建并归一化 Train/Test 张量 + 注入 Grid Encoding ...")
        train_x, train_y, test_x, test_y = _build_and_normalize_tensors(
            ds_map=ds_map,
            axis=axis,
            n_total_take=n_total_take,
            n_train=n_train,
            n_test=n_test,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            target_res=target_res,
            log_every=50,
        )

    train_path = f"{out_dir.rstrip('/')}/cfd2d_train_{target_res}.pt"
    test_path = f"{out_dir.rstrip('/')}/cfd2d_test_{target_res}.pt"
    stats_path = f"{out_dir.rstrip('/')}/cfd2d_norm_stats_{target_res}.pt"

    print(f"正在保存 {train_path}... Shape: x {list(train_x.shape)}, y {list(train_y.shape)}")
    torch.save({"x": train_x, "y": train_y}, train_path)

    print(f"正在保存 {test_path}... Shape: x {list(test_x.shape)}, y {list(test_y.shape)}")
    torch.save({"x": test_x, "y": test_y}, test_path)

    print(f"正在保存 {stats_path}... Shape: y_mean {list(y_mean.shape)}, y_std {list(y_std.shape)}")
    torch.save({"y_mean": y_mean, "y_std": y_std}, stats_path)

    print("完成。")


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess PDEBench CFD-2D HDF5 to FNO-ready .pt files.")
    p.add_argument("--h5", type=str, required=True, help="PDEBench CFD-2D 原始 HDF5 文件路径")
    p.add_argument("--out-dir", type=str, default=".", help="输出目录")
    p.add_argument("--target-res", type=int, default=64, help="目标导出分辨率(默认64)，支持从原始128/512下采样")

    p.add_argument("--n-total", type=int, default=2000, help="期望导出的总样本数(默认2000)")
    p.add_argument("--n-train", type=int, default=1800, help="训练样本数(默认1800)")
    p.add_argument("--n-test", type=int, default=200, help="测试样本数(默认200)")

    p.add_argument("--eps", type=float, default=1e-6, help="std 防除零 eps")

    p.add_argument("--density-path", type=str, default=None, help="手动指定 Density dataset 路径")
    p.add_argument("--pressure-path", type=str, default=None, help="手动指定 Pressure dataset 路径")
    p.add_argument("--vx-path", type=str, default=None, help="手动指定 Vx dataset 路径")
    p.add_argument("--vy-path", type=str, default=None, help="手动指定 Vy dataset 路径")

    args = p.parse_args()

    preprocess_pdebench_cfd2d_to_pt(
        h5_path=args.h5,
        out_dir=args.out_dir,
        target_res=args.target_res,
        n_total=args.n_total,
        n_train=args.n_train,
        n_test=args.n_test,
        eps=args.eps,
        density_path=args.density_path,
        pressure_path=args.pressure_path,
        vx_path=args.vx_path,
        vy_path=args.vy_path,
    )


if __name__ == "__main__":
    main()