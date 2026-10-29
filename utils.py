"""Data, fingerprints, plots and the assignment report.

Only the standard library is imported here.  CUDA policy is established by
``baseline.py`` before any function imports PyTorch or another numeric library.
"""
import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGES = ("scratch", "native", "d4")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Write a complete JSON file atomically, preserving literal newlines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def paired_paths(root):
    """Return sorted PNG pairs in one split, rejecting unmatched filenames."""
    root = Path(root)
    inputs = sorted((root / "input").glob("*.png"))
    targets = sorted((root / "target").glob("*.png"))
    if not inputs:
        raise ValueError(f"No input PNG files found in {root / 'input'}")
    input_names, target_names = {p.name for p in inputs}, {p.name for p in targets}
    if input_names != target_names:
        raise ValueError(f"Unmatched input/target PNG names in {root}: "
                         f"missing targets={sorted(input_names-target_names)[:8]}, "
                         f"missing inputs={sorted(target_names-input_names)[:8]}")
    return [(path, root / "target" / path.name) for path in inputs]


def fingerprint_data(data):
    """Hash all supplied pairs, independent of this checkout's absolute path.

    ``files_sha256`` is the SHA-256 of the canonical JSON mapping from relative
    PNG filenames to each file's SHA-256.  It can be pinned in ``config.json``
    without shipping a separate 2,640-entry manifest.
    """
    data = Path(data).resolve()
    files, counts = {}, {}
    for split, expected_count in (("train", 1200), ("val", 120)):
        pairs = paired_paths(data / split)
        if len(pairs) != expected_count:
            raise ValueError(f"Expected {expected_count} {split} pairs; found {len(pairs)}")
        counts[split] = len(pairs)
        for pair in pairs:
            for path in pair:
                files[path.relative_to(data).as_posix()] = sha256(path)
    return {"train_pairs": counts["train"], "val_pairs": counts["val"],
            "verified_files": len(files), "files_sha256": object_sha(files)}


def make_dataset(root, patch=0, augment=False, cache=True):
    """Build the paired dataset; every item is ``(input, target, index)``.

    Images are RGB uint8 while cached and become CHW FP32 / 255 after cropping
    and augmentation.  Crop top, crop left, quarter rotation and horizontal
    flip consume the global Torch RNG in that exact order, matching training.
    ``paths`` exposes sorted input paths; ``images`` contains cached pairs (or
    is ``None`` when caching is disabled). No DataLoader is created here.
    """
    import numpy as np
    import torch
    from PIL import Image

    class PairedImages:
        def __init__(self):
            self.pairs = paired_paths(root)
            self.paths = [pair[0] for pair in self.pairs]
            self.patch, self.augment = int(patch), bool(augment)
            if self.patch < 0:
                raise ValueError("patch must be non-negative")
            self.images = [self._read(pair) for pair in self.pairs] if cache else None

        def _read(self, paths):
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(np.asarray(image.convert("RGB")).copy())
            if images[0].shape != images[1].shape:
                raise ValueError(f"Input/target image shapes differ: {paths[0].name}")
            if self.patch and min(images[0].shape[:2]) < self.patch:
                raise ValueError(f"Image is smaller than patch {self.patch}: {paths[0].name}")
            return tuple(images)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            x, y = self.images[index] if self.images is not None else self._read(self.pairs[index])
            if self.patch:
                height, width = x.shape[:2]
                top = int(torch.randint(height - self.patch + 1, ()))
                left = int(torch.randint(width - self.patch + 1, ()))
                x = x[top:top+self.patch, left:left+self.patch]
                y = y[top:top+self.patch, left:left+self.patch]
            if self.augment:
                turns = int(torch.randint(4, ()))
                x, y = np.rot90(x, turns), np.rot90(y, turns)
                if bool(torch.randint(2, ())):
                    x, y = np.fliplr(x), np.fliplr(y)
            x, y = (torch.from_numpy(a.transpose(2, 0, 1).copy()).float() / 255 for a in (x, y))
            return x, y, index

    return PairedImages()


def read_histories(run_dir):
    """Read the three stages' loss/PSNR histories without numeric imports."""
    run_dir, result = Path(run_dir), {}
    for stage in STAGES:
        path = run_dir / stage / "history.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                result[stage] = [{key: float(value) for key, value in row.items()}
                                 for row in csv.DictReader(stream)]
    if not result:
        raise ValueError(f"No stage history.csv files found below {run_dir}")
    return result


def _draw_curves(axes, histories):
    """Plot each stage on its own epoch axis; never join different predictors."""
    titles = {"scratch": "Scratch: single view", "native": "Native: single view",
              "d4": "D4: eight-view mean"}
    for axis, stage in zip(axes, STAGES):
        rows = histories.get(stage, [])
        axis.set_title(titles[stage], fontsize=10)
        if not rows:
            axis.text(0.5, 0.5, "Not run", ha="center", va="center", transform=axis.transAxes)
            continue
        epochs = [row["epoch"] for row in rows]
        axis.plot(epochs, [row["train_loss"] for row in rows], color="#2870ac",
                  linewidth=1.0, label="Train (online, crops)")
        axis.plot(epochs, [row["val_loss"] for row in rows], color="#d77b20",
                  linewidth=1.3, linestyle="--", label="Validation (EMA, full)")
        if len(epochs) > 1:
            axis.set_xlim(epochs[0], epochs[-1])
        axis.set_xlabel("Epoch within stage", fontsize=9)
        axis.set_ylabel("Mean per-image ln(MSE)", fontsize=9)
        axis.legend(fontsize=7, loc="best")
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=8)


def plot_history(run_dir, output):
    """Save assignment train/val curves from existing logs only."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.1), constrained_layout=True)
    _draw_curves(axes, read_histories(run_dir))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return str(output)


def _report_metrics(run_dir):
    run_dir = Path(run_dir)
    for path in (run_dir / "evaluation.json", run_dir / "evaluation" / "metrics.json",
                 run_dir / "metrics.json"):
        if path.exists():
            return read_json(path)
    raise ValueError("Report requires an existing full-validation evaluation JSON")


def make_report(run_dir, data, onnx_path, output, challenge=None, metrics=None, session=None):
    """Make the required two-page PDF and two comparison PNGs.

    This is an explicit CPU inference operation: two validation examples are
    restored for the figures. It never trains a model or uses CUDA. A checked
    session and full-validation metrics can be supplied by the caller to reuse
    the evaluation; otherwise the unchanged official checker builds a session.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    import numpy as np
    from PIL import Image
    from onnx_io import check, run_session

    run_dir, data, onnx_path, output = map(Path, (run_dir, data, onnx_path, output))
    spec = read_json(ROOT / "challenge.json") if challenge is None else (
        read_json(challenge) if isinstance(challenge, (str, Path)) else challenge)
    metrics = _report_metrics(run_dir) if metrics is None else metrics
    if metrics.get("onnx_sha256") and metrics["onnx_sha256"] != sha256(onnx_path):
        raise ValueError("Report metrics do not belong to this ONNX file")
    if int(metrics.get("validation_pairs", 0)) != 120:
        raise ValueError("Report requires evaluation of all 120 validation pairs")
    if session is None:
        stored_elements, session = check(onnx_path, spec, announce=False)
    else:
        stored_elements = metrics.get("stored_onnx_elements")
        if stored_elements is None:
            raise ValueError("When reusing a checked session, supply stored_onnx_elements")

    histories = read_histories(run_dir)
    configs, results = {}, {}
    for stage in STAGES:
        for mapping, filename in ((configs, "config.json"), (results, "result.json")):
            path = run_dir / stage / filename
            if path.exists():
                mapping[stage] = read_json(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    restored_score = float(metrics["restored_psnr_mean"])
    input_score = float(metrics["input_psnr_mean"])
    parameters = metrics.get("trainable_parameters")
    parameter_label = f"{int(parameters):,}" if parameters is not None else "not recorded"
    strict_profile = configs.get("d4", {}).get("profile", "strict") == "strict"
    profile_description = (
        "Strict profile: deterministic kernels, TF32/compile off; independent seeded shuffle sampler."
        if strict_profile else
        "Historical profile: original global-RNG sampler and compiled training; not bitwise deterministic."
    )
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.07, 0.965, "Image restoration: shared NAFNet + D4 fine-tuning",
                fontsize=14, weight="bold", va="top")
    method = (
        "Network: residual encoder-decoder, widths 32/64/128; encoder 2/2, middle 7, decoder 2/2.\n"
        "Convolutions are shared within each stage; normalization and residual scales are independent.\n"
        "D4: the same network processes 8 rotations/reflections; inverse-transform and average outputs.\n"
        "D4 fine-tuning backpropagates through this average; no additional trainable parameters.\n"
        "Data: supplied 1,200 training pairs only; 256x256 crops, random quarter rotations and flips.\n"
        "Validation: 120 supplied full 384x384 RGB pairs; clean validation images never enter training.\n"
        "Loss: batch mean of ln(max(per-image MSE, 1e-12)); raw outputs during training.\n"
        "AdamW (betas 0.9/0.999, eps 1e-8, weight decay 1e-4), gradient norm clip 1, EMA 0.999.\n"
        "At each new stage: load previous EMA weights; reset optimizer, scheduler and seeded RNG.\n"
        + profile_description
    )
    figure.text(0.07, 0.925, method, fontsize=8.2, linespacing=1.48, va="top")
    table_axis = figure.add_axes([0.065, 0.60, 0.87, 0.12])
    table_axis.axis("off")
    table_rows = []
    for stage in STAGES:
        config, result = configs.get(stage, {}), results.get(stage, {})
        rows = histories.get(stage, [])
        best_psnr = result.get("best_val_psnr", max((r["val_psnr"] for r in rows), default=float("nan")))
        lr = config.get("learning_rate")
        min_lr = config.get("min_learning_rate")
        table_rows.append([
            stage, str(config.get("epochs", len(rows))), str(config.get("cosine_t_max", "?")),
            f"{lr:g} -> {min_lr:g}" if lr is not None and min_lr is not None else "see config",
            str(config.get("batch_size", "?")), str(config.get("seed", "?")),
            "BF16" if config.get("amp_dtype") == "bfloat16" else "FP32", f"{best_psnr:.4f}"
        ])
    table = table_axis.table(cellText=table_rows,
                             colLabels=["Stage", "Epochs", "Cosine Tmax", "LR -> minimum", "Batch", "Seed", "Precision", "Best val"],
                             loc="center", cellLoc="center", colWidths=[.10, .09, .13, .23, .08, .08, .11, .18])
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1, 1.65)
    final = results.get("d4", {})
    figure.text(0.07, 0.588,
                f"Final: {parameter_label} trainable parameters; {stored_elements:,}/{spec['max_model_elements']:,} stored ONNX elements.\n"
                f"Official CPU ONNX PSNR: {input_score:.4f} dB input -> {restored_score:.4f} dB restored (all 120).\n"
                f"Selected D4 checkpoint: epoch {final.get('best_epoch', '?')}; clipping [0,1] is inside ONNX.\n"
                "Mean of per-image full-image RGB PSNR; no border crop or additional external D4.",
                fontsize=8.3, linespacing=1.55, va="top")
    curve_axes = [figure.add_axes([left, 0.26, 0.23, 0.21]) for left in (0.10, 0.41, 0.72)]
    _draw_curves(curve_axes, histories)
    figure.text(0.07, 0.195,
                "Each panel uses local epochs and its own vertical range; no smoothing or missing epochs.\n"
                "D4 changes the predictor to an eight-view mean; stages are not joined across that change.",
                fontsize=7.2, linespacing=1.45, va="top")
    figure.text(0.07, 0.135,
                "References: Chen et al., Simple Baselines for Image Restoration (ECCV 2022, NAFNet).\n"
                "Lim et al., Enhanced Deep Residual Networks for Single Image Super-Resolution (CVPRW 2017).\n"
                "Shared convolutions and training through the D4 average are project adaptations; no novelty claim.\n"
                "Model selection used this validation split; hidden-scene PSNR is not measured here.",
                fontsize=7.6, linespacing=1.5, va="top")

    pairs = paired_paths(data / "val")
    selected = [pairs[0], pairs[len(pairs) // 2]]
    score_map = {row["name"]: row for row in metrics.get("per_image", [])}
    examples = []
    for input_path, target_path in selected:
        with Image.open(input_path) as image:
            degraded = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        with Image.open(target_path) as image:
            clean = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        prediction = run_session(session, degraded.transpose(2, 0, 1)[None])[0].transpose(1, 2, 0)
        examples.append((input_path.name, degraded, prediction, clean))

    def draw_examples(comparison, examples_to_draw):
        count = len(examples_to_draw)
        grid = comparison.subplots(count * 2, 3, squeeze=False)
        for index, (name, degraded, prediction, clean) in enumerate(examples_to_draw):
            height, width = clean.shape[:2]
            size = min(96, height, width)
            top, left = (height-size)//2, (width-size)//2
            scores = score_map.get(name, {})
            labels = ("Degraded input", "Restored (D4)", "Clean target")
            for column, (image, label) in enumerate(zip((degraded, prediction, clean), labels)):
                upper, lower = grid[2*index, column], grid[2*index+1, column]
                upper.imshow(np.clip(image, 0, 1))
                upper.add_patch(Rectangle((left, top), size, size, fill=False, edgecolor="#ff2f4b", linewidth=1.2))
                title = label
                score_key = "input_psnr" if column == 0 else "restored_psnr" if column == 1 else None
                if score_key in scores:
                    title += f" ({scores[score_key]:.2f} dB)"
                upper.set_title(f"{name}  {title}", fontsize=7.5)
                lower.imshow(np.clip(image[top:top+size, left:left+size], 0, 1), interpolation="nearest")
                lower.set_title("Same central 96x96 zoom", fontsize=7)
                upper.axis("off")
                lower.axis("off")

    page2 = plt.figure(figsize=(8.27, 11.69))
    page2.suptitle("Validation examples: full images and matching zooms", fontsize=13, y=0.97)
    draw_examples(page2, examples)
    page2.subplots_adjust(left=.045, right=.955, bottom=.04, top=.92, hspace=.28, wspace=.10)
    with PdfPages(output) as pdf:
        info = pdf.infodict()
        info["Title"] = "Image Restoration Challenge Report"
        info["Subject"] = "Three-stage shared NAFNet and D4 fine-tuning"
        pdf.savefig(figure)
        pdf.savefig(page2)
    plt.close(figure)
    plt.close(page2)
    for index, example in enumerate(examples, 1):
        comparison = plt.figure(figsize=(9, 6))
        draw_examples(comparison, [example])
        comparison.subplots_adjust(left=.02, right=.98, bottom=.02, top=.95, hspace=.18, wspace=.08)
        comparison.savefig(run_dir / f"comparison_{index}.png", dpi=160)
        plt.close(comparison)
    plot_history(run_dir, run_dir / "loss.png")
    return {"report": str(output), "pages": 2, "examples": [p[0].name for p in selected],
            "onnx_sha256": sha256(onnx_path), "restored_psnr_mean": restored_score}
