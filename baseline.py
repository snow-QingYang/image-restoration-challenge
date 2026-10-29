"""Assignment training, ONNX export, evaluation and report generation.

Original README commands work unchanged. ``train`` runs scratch -> native ->
D4 from random initialization; use a NEW --run directory to keep the delivered
weights. Example: python baseline.py train --run runs/retrain --gpu 0

``train --check-only`` checks the untrained final model on CPU without training.
``train --dry-run`` only prints all three training configurations. No training
or GPU access occurs when importing this file or running export/evaluate/report.
"""
import argparse
import copy
import csv
from contextlib import nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time

from utils import (ROOT, STAGES, fingerprint_data, make_dataset, make_report,
                   paired_paths, read_json, sha256, write_json)


def configuration(path, stage=None):
    config = read_json(path)
    if stage:
        overrides = config.pop("stages")[stage]
        config.update(overrides)
        config["stage"] = stage
        config.pop("reference_environment", None)
        config.pop("recipe_notes", None)
        if (config["profile"] != "strict" or not config["deterministic_algorithms"]
                or not config["independent_sampler"] or config["compile_train"]):
            raise ValueError("This training route uses strict deterministic eager execution")
        if config["workers"] != 0:
            raise ValueError("The reproducible crop/RNG order requires workers=0")
        if config["loss"] != "mean_log_per_image_mse":
            raise ValueError("Supported loss: mean_log_per_image_mse")
        log_every = config.get("log_every_batches", 10)
        if type(log_every) is not int or log_every < 1:
            raise ValueError("log_every_batches must be a positive integer")
    return config


def code_fingerprints():
    return {name: sha256(ROOT / name) for name in ("model.py", "baseline.py", "utils.py")}


def prepare_environment(args, config):
    # Set Python hash seed at interpreter startup, before scientific imports.
    updates = {"PYTHONHASHSEED": str(config["seed"]), "CUDA_VISIBLE_DEVICES": str(args.gpu),
               "OMP_NUM_THREADS": str(config["threads"]), "MKL_NUM_THREADS": str(config["threads"]),
               "OPENBLAS_NUM_THREADS": str(config["threads"]),
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": "0"}
    if any(os.environ.get(key) != value for key, value in updates.items()):
        environment = os.environ.copy()
        environment.update(updates)
        os.execve(sys.executable, [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]], environment)


def execute(args, config):
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite {args.output}; use a new --run even for --resume")
    prepare_environment(args, config)
    # Validate input bytes before allocating a model or touching CUDA.
    data_binding = fingerprint_data(args.data)
    if data_binding['files_sha256'] != config['data_files_sha256']:
        raise ValueError('Dataset bytes differ from the supplied assignment split')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, RandomSampler
    from model import build_model, load_ema_weights

    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    strict = args.profile == "strict"
    torch.use_deterministic_algorithms(strict, warn_only=False)
    if strict:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise ValueError("An available CUDA GPU is required; no CPU fallback")
    device = torch.device("cuda:0")  # explicit physical card is isolated by CUDA_VISIBLE_DEVICES
    if config["amp_dtype"] and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("D4 training requires native CUDA BF16 support")

    # Each phase resets RNG and constructs the model, even if its random weights
    # will immediately be replaced by the previous phase's EMA.
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(config)
    initialization = {"mode": "random", "seed": seed}
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume.get("schema") != "assignment.epoch.v1" or resume.get("batch_cursor") != 0:
            raise ValueError("Legacy/best weights cannot resume: use a new numbered/last full checkpoint. "
                             "Legacy weights can only initialize the next phase via --init-checkpoint.")
    elif args.init_checkpoint:
        source = args.init_checkpoint.resolve()
        initial = torch.load(source, map_location="cpu", weights_only=True)
        if initial.get("epoch") != config["source_epoch"]:
            raise ValueError(f"{args.stage} requires {config['source_stage']} epoch {config['source_epoch']}; "
                             f"received epoch {initial.get('epoch')}")
        source_stage = initial.get("stage")
        if source_stage and source_stage != config["source_stage"]:
            raise ValueError(f"Wrong source stage: {source_stage}")
        if initial.get("weights_source") != "ema":
            raise ValueError("Stage transition requires EMA, not online weights")
        details = load_ema_weights(model, initial["model"])
        initialization = {"mode": "previous_stage_ema_only", "checkpoint": str(source),
                          "sha256": sha256(source), "source_stage": config["source_stage"],
                          "source_epoch": initial["epoch"], "details": details,
                          "optimizer_scheduler_rng": "reset, not resumed"}

    train_data = make_dataset(args.data / "train", config["patch_size"], True, config["cache_images"])
    val_data = make_dataset(args.data / "val", cache=config["cache_images"])
    sampler_generator = torch.Generator(device="cpu").manual_seed(seed)
    sampler = RandomSampler(train_data, generator=sampler_generator)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], sampler=sampler,
                              num_workers=config["workers"], pin_memory=True, drop_last=False)
    # Keep a real validation DataLoader. Even workers=0 iterators consume CPU RNG
    # base_seed; replacing this with direct dataset iteration changes history.
    val_loader = DataLoader(val_data, batch_size=1, num_workers=0)
    model = model.to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    training_model = model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], betas=tuple(config["betas"]),
                                  eps=config["eps"], amsgrad=False, maximize=False, capturable=False,
                                  differentiable=False, foreach=config["optimizer_foreach"],
                                  fused=config["optimizer_fused"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["cosine_t_max"], eta_min=config["min_learning_rate"])

    def environment_record():
        properties = torch.cuda.get_device_properties(device)
        dependencies = {}
        for name in ("torch", "numpy", "Pillow", "onnx", "onnxruntime", "triton"):
            try:
                dependencies[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                dependencies[name] = None
        return {"python": platform.python_version(), "platform": platform.platform(),
                "packages": dependencies, "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
                "gpu": {"name": properties.name, "capability": [properties.major, properties.minor],
                        "total_memory": properties.total_memory,
                        "multiprocessor_count": properties.multi_processor_count},
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "pythonhashseed": os.environ["PYTHONHASHSEED"]}

    bindings = {"config": config, "code": code_fingerprints(), "data": data_binding,
                "environment": environment_record()}

    def rng_state():
        state = np.random.get_state()
        return {"python": random.getstate(),
                "numpy": {"kind": state[0], "keys": state[1].tolist(), "position": state[2],
                          "has_gauss": state[3], "cached_gaussian": state[4]},
                "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}

    def restore_rng(state):
        random.setstate(state["python"])
        n = state["numpy"]
        np.random.set_state((n["kind"], np.asarray(n["keys"], dtype=np.uint32), n["position"],
                             n["has_gauss"], n["cached_gaussian"]))
        torch.set_rng_state(state["torch_cpu"])
        torch.cuda.set_rng_state_all(state["torch_cuda"])

    def cpu_state(module):
        return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}

    def save_checkpoint(path, value):
        temporary = path.with_name(path.name + ".tmp")
        torch.save(value, temporary)
        os.replace(temporary, path)

    def log_mse(prediction, target):
        error = (prediction - target).square().flatten(1).mean(1)
        return error.clamp_min(1e-12).log().mean()

    def validate():
        ema.eval()
        losses, scores, images = 0., [], []
        with torch.inference_mode():
            for x, y, indices in val_loader:
                x, y = x.to(device), y.to(device)
                prediction = ema(x)  # FP32 eager; no autocast or outside D4
                losses += float(log_mse(prediction, y))
                error = (prediction.detach().cpu().double().clamp(0, 1)
                         - y.detach().cpu().double()).square().flatten(1).mean(1)
                score = float((-10 * error.clamp_min(1e-12).log10())[0])
                scores.append(score)
                images.append({"name": val_data.paths[int(indices[0])].name, "psnr": score})
        return {"loss": losses / len(val_data), "psnr": float(np.mean(scores)), "images": images,
                "metric": "FP32 target -> CPU float64 RGB; prediction clamp[0,1]; full120 mean PSNR"}

    completed, global_step, best_epoch = 0, 0, None
    best_score, best_model, best_validation = None, None, None
    rows = []
    if resume:
        if resume["bindings"] != bindings:
            raise ValueError("Resume config/code/data/environment differ from the saved boundary; refusing")
        if not 0 < resume["epoch_completed"] < config["epochs"]:
            raise ValueError("Resume must be an unfinished completed-epoch boundary")
        model.load_state_dict(resume["online_model"], strict=True)
        ema.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        sampler_generator.set_state(resume["sampler_state"])
        completed, global_step = resume["epoch_completed"], resume["global_step"]
        best_score, best_epoch = resume["best_val_psnr"], resume["best_epoch"]
        best_model, best_validation = resume["best_model"], resume["best_validation"]
        rows, initialization = resume["history"], resume["initialization"]
        # Restore only after model/optimizer/loader construction and data IO.
        restore_rng(resume["RNG"])

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "checkpoints").mkdir()
    write_json(args.output / "config.json", config)
    write_json(args.output / "bindings.json", bindings)
    write_json(args.output / "initialization.json", initialization)
    write_json(args.output / "run.json", {"stage": args.stage, "profile": args.profile,
               "data": str(args.data.resolve()), "physical_gpu": args.gpu,
               "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
               "resume_sha256": sha256(args.resume) if args.resume else None,
               "resume_completed_epoch": completed,
               "parameters": sum(p.numel() for p in model.parameters()),
               "score_status": "This invocation measures its own result; historical scores are not a guarantee"})

    def compact_best():
        return {"model": best_model, "config": config, "stage": args.stage, "epoch": best_epoch,
                "val_psnr": best_score, "val_loss": best_validation["loss"], "weights_source": "ema",
                "profile": args.profile, "resume_support": "use numbered or last full checkpoint"}

    if resume:
        save_checkpoint(args.output / "best.pt", compact_best())
    fields = ["epoch", "train_loss", "val_loss", "val_psnr", "learning_rate_used", "learning_rate_next"]
    log_every = config.get("log_every_batches", 10)
    started = time.perf_counter()
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as history_stream, \
         (args.output / "steps.jsonl").open("w", encoding="utf-8") as step_stream:
        writer = csv.DictWriter(history_stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        history_stream.flush()
        for epoch in range(completed + 1, config["epochs"] + 1):
            model.train()
            total, order = 0., []
            learning_rate_used = optimizer.param_groups[0]["lr"]
            for batch_index, (x, y, indices) in enumerate(train_loader, 1):
                names = [train_data.paths[int(i)].name for i in indices]
                order.extend(names)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if config["amp_dtype"] else nullcontext()
                with amp:
                    prediction = training_model(x)
                    loss = log_mse(prediction.float() if config["amp_dtype"] else prediction, y)
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"Nonfinite loss at epoch {epoch}, batch {batch_index}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"],
                                               norm_type=2., error_if_nonfinite=False,
                                               foreach=config["clip_foreach"])
                optimizer.step()
                with torch.no_grad():
                    for average, current in zip(ema.parameters(), model.parameters()):
                        average.lerp_(current.detach(), 1. - config["ema_decay"])
                    for average, current in zip(ema.buffers(), model.buffers()):
                        average.copy_(current)
                loss_value = float(loss.detach())
                total += loss_value * len(x)
                global_step += 1
                step_stream.write(json.dumps({"epoch": epoch, "batch": batch_index, "step": global_step,
                                              "names": names, "loss": loss_value,
                                              "loss_hex": loss_value.hex()}) + "\n")
                if batch_index == 1 or batch_index % log_every == 0 or batch_index == len(train_loader):
                    step_stream.flush()
                    print(f"{args.stage}: epoch {epoch}/{config['epochs']}, "
                          f"batch {batch_index}/{len(train_loader)} | "
                          f"loss {loss_value:.6f} | mean loss {total / len(order):.6f} | "
                          f"lr {learning_rate_used:.10g}", flush=True)
            result = validate()
            scheduler.step()  # once, AFTER this epoch's EMA validation
            row = {"epoch": epoch, "train_loss": total / len(train_data), "val_loss": result["loss"],
                   "val_psnr": result["psnr"], "learning_rate_used": learning_rate_used,
                   "learning_rate_next": optimizer.param_groups[0]["lr"]}
            rows.append(row)
            writer.writerow(row)
            history_stream.flush()
            step_stream.flush()
            is_best = best_score is None or result["psnr"] > best_score
            if is_best:
                best_score, best_epoch = result["psnr"], epoch
                best_model, best_validation = cpu_state(ema), result
                save_checkpoint(args.output / "best.pt", compact_best())
            write_json(args.output / "validation" / f"epoch_{epoch:04d}.json", result)
            write_json(args.output / "orders" / f"epoch_{epoch:04d}.json", order)
            checkpoint = {"schema": "assignment.epoch.v1", "batch_cursor": 0,
                          "model": cpu_state(ema), "online_model": cpu_state(model),
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "RNG": rng_state(),
                          "sampler_state": sampler_generator.get_state() if strict else None,
                          "stage": args.stage, "profile": args.profile, "config": config,
                          "epoch": epoch, "epoch_completed": epoch, "global_step": global_step,
                          "val_psnr": result["psnr"], "val_loss": result["loss"], "weights_source": "ema",
                          "best_val_psnr": best_score, "best_epoch": best_epoch,
                          "best_model": best_model, "best_validation": best_validation,
                          "initialization": initialization, "history": rows, "bindings": bindings}
            save_checkpoint(args.output / "checkpoints" / f"epoch_{epoch:04d}.pt", checkpoint)
            save_checkpoint(args.output / "last.pt", checkpoint)
            print(f"{args.stage} {epoch}/{config['epochs']} | "
                  f"train loss {row['train_loss']:.6f} | val loss {row['val_loss']:.6f} | "
                  f"EMA val PSNR {result['psnr']:.8f} | "
                  f"lr used {learning_rate_used:.10g}, next {row['learning_rate_next']:.10g}", flush=True)
    write_json(args.output / "result.json", {"stage": args.stage, "profile": args.profile,
               "completed_epochs": config["epochs"], "best_epoch": best_epoch,
               "best_val_psnr": best_score, "best_checkpoint": "best.pt",
               "last_checkpoint": "last.pt", "seconds_this_invocation": time.perf_counter() - started,
               "metric": best_validation["metric"], "initialization": initialization})



def export_model(args, untrained=False):
    import torch
    from model import build_model, load_ema_weights
    from onnx_io import check

    torch.set_num_threads(4)
    if untrained:
        config = configuration(args.config)
        model = build_model(config)
    else:
        checkpoint_path = args.checkpoint or args.run / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        model = build_model(config)
        load_ema_weights(model, checkpoint["model"])
    model = model.cpu().eval()
    wrapper = torch.nn.Sequential(model, torch.nn.Hardtanh(0, 1)).eval()
    spec = read_json(ROOT / "challenge.json")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(wrapper, torch.zeros(1, 3, spec["image_size"], spec["image_size"]),
                          str(args.output), dynamo=False, input_names=["input"],
                          output_names=["output"], opset_version=spec["opset"])
    elements, _ = check(args.output, spec)
    print(f"Trainable parameters: {parameters:,}; stored ONNX elements: {elements:,}", flush=True)
    if not untrained:
        write_json(args.run / "export.json", {"checkpoint": str(checkpoint_path.resolve()),
                   "checkpoint_sha256": sha256(checkpoint_path), "epoch": checkpoint["epoch"],
                   "weights_source": checkpoint.get("weights_source", "model"),
                   "trainable_parameters": parameters, "stored_onnx_elements": elements,
                   "onnx": str(args.output.resolve()), "onnx_sha256": sha256(args.output)})
    return elements


def check_only(args):
    # Independent process during real training: preflight never advances its RNG.
    with tempfile.TemporaryDirectory(prefix="assignment-check-") as temporary:
        args.output = Path(temporary) / "model.onnx"
        export_model(args, untrained=True)
    print("Untrained model passed the official CPU checks; no training performed.", flush=True)


def evaluate(args):
    import math
    from onnx_io import check, score_pair

    spec = read_json(ROOT / "challenge.json")
    elements, session = check(args.model, spec)
    rows = []
    for index, (input_path, target_path) in enumerate(paired_paths(args.data / "val"), 1):
        raw, restored = score_pair(session, input_path, target_path)
        rows.append({"name": input_path.name, "input_psnr": raw, "restored_psnr": restored})
        if index % 10 == 0:
            print(f"CPU ONNX validation: {index} pairs", flush=True)
    metrics = {"validation_pairs": len(rows),
               "input_psnr_mean": math.fsum(row["input_psnr"] for row in rows) / len(rows),
               "restored_psnr_mean": math.fsum(row["restored_psnr"] for row in rows) / len(rows),
               "stored_onnx_elements": elements,
               "onnx_sha256": sha256(args.model), "per_image": rows,
               "metric": "Official CPU ONNX; mean of full-image RGB per-image PSNR; PNG target float64 /255"}
    export_record = args.run / "export.json"
    if export_record.exists():
        record = read_json(export_record)
        if record["onnx_sha256"] == metrics["onnx_sha256"]:
            metrics["trainable_parameters"] = record["trainable_parameters"]
    write_json(args.output or args.run / "evaluation.json", metrics)
    print(f"{len(rows)} validation pairs: input {metrics['input_psnr_mean']:.8f} dB -> "
          f"restored {metrics['restored_psnr_mean']:.8f} dB", flush=True)
    return metrics, session


def report(args):
    result = make_report(args.run, args.data, args.model, args.output)
    print(json.dumps(result, indent=2), flush=True)


def train(args):
    full_config = configuration(args.config)
    args.gpu = full_config["gpu"] if args.gpu is None else args.gpu
    if args.gpu < 0:
        raise ValueError("--gpu must be a nonnegative physical GPU index")
    if args.check_only:
        check_only(args)
        return
    stages = STAGES if args.stage == "all" else (args.stage,)
    if args.stage == "all" and (args.init_checkpoint or args.resume):
        raise ValueError("--init-checkpoint/--resume require one explicit --stage")
    if args.dry_run:
        print(json.dumps({"action": "PLAN ONLY", "data": str(args.data), "run": str(args.run),
                          "physical_gpu": args.gpu,
                          "stages": {stage: configuration(args.config, stage) for stage in stages},
                          "transition": " -> ".join([
                              f"{full_config['stages'][stage]['source_stage']} epoch "
                              f"{full_config['stages'][stage]['source_epoch']} EMA"
                              for stage in ("native", "d4")
                          ] + ["D4 best EMA"])}, indent=2))
        return
    for stage in stages:
        if (args.run / stage).exists():
            raise ValueError(f"Refusing to overwrite {args.run / stage}; select a new --run")
    if args.stage == "all" and (args.run / "best.pt").exists():
        raise ValueError("Refusing to overwrite delivered best.pt; select a new --run")
    if not args._stage_worker:
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "train", "--check-only",
                        "--config", str(args.config)], check=True, cwd=ROOT)
    if args.stage == "all":
        for stage in STAGES:
            config = configuration(args.config, stage)
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "train", "--stage", stage,
                       "--config", str(args.config), "--run", str(args.run), "--data", str(args.data),
                       "--gpu", str(args.gpu), "--_stage-worker"]
            if config["source_stage"]:
                source = args.run / config["source_stage"] / "checkpoints" / f"epoch_{config['source_epoch']:04d}.pt"
                command.extend(["--init-checkpoint", str(source)])
            subprocess.run(command, check=True, cwd=ROOT)
        write_json(args.run / "config.json", full_config)
        shutil.copy2(args.run / "d4" / "best.pt", args.run / "best.pt")
        # Outputs of explicit `train`, in addition to weights and per-epoch logs.
        destination = args.run / "submission"
        for command in (
            ["export", "--output", str(destination / "model.onnx")],
            ["evaluate", str(destination / "model.onnx"), "--data", str(args.data)],
            ["report", "--model", str(destination / "model.onnx"), "--data", str(args.data),
             "--output", str(destination / "report.pdf")],
        ):
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *command,
                            "--run", str(args.run)], check=True, cwd=ROOT)
        return
    if args.stage == "scratch" and args.init_checkpoint:
        raise ValueError("scratch must start from random initialization")
    if args.stage != "scratch" and not (args.init_checkpoint or args.resume):
        raise ValueError("native/d4 require --init-checkpoint (previous EMA) or --resume (same stage)")
    args.output = args.run / args.stage
    args.profile = "strict"
    execute(args, configuration(args.config, args.stage))
    if args.stage == "d4":
        shutil.copy2(args.output / "best.pt", args.run / "best.pt")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("train", "Train all three stages from scratch"),
                            ("export", "Export EMA checkpoint and run official CPU checks"),
                            ("evaluate", "Evaluate ONNX on every validation pair using official metrics"),
                            ("report", "Generate the two-page assignment report and comparison figures")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--run", type=Path, default=ROOT / "runs" / "baseline")
        if name in ("train", "export"):
            command.add_argument("--config", type=Path, default=ROOT / "config.json")
        if name in ("train", "evaluate", "report"):
            command.add_argument("--data", type=Path, default=ROOT / "data")
        if name == "train":
            command.add_argument("--gpu", type=int, help="Physical CUDA index; defaults to config.json")
            command.add_argument("--stage", choices=("all", *STAGES), default="all")
            command.add_argument("--check-only", action="store_true", help="Only check untrained final model on CPU")
            command.add_argument("--dry-run", action="store_true", help="Print recipe only; no model/data/GPU work")
            source = command.add_mutually_exclusive_group()
            source.add_argument("--init-checkpoint", type=Path)
            source.add_argument("--resume", type=Path, help="Full completed-epoch checkpoint from this trainer")
            command.add_argument("--_stage-worker", action="store_true", help=argparse.SUPPRESS)
        elif name == "export":
            command.add_argument("--checkpoint", type=Path, help="Default: RUN/best.pt")
            command.add_argument("--output", type=Path, default=ROOT / "submission" / "model.onnx")
        elif name == "evaluate":
            command.add_argument("model", nargs="?", type=Path, default=ROOT / "submission" / "model.onnx")
            command.add_argument("--output", type=Path, help="Default: RUN/evaluation.json")
        else:
            command.add_argument("--model", type=Path, default=ROOT / "submission" / "model.onnx")
            command.add_argument("--output", type=Path, default=ROOT / "submission" / "report.pdf")
    args = parser.parse_args()
    # Child processes run from ROOT; resolve user-relative paths before spawning.
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


def main():
    args = arguments()
    {"train": train, "export": export_model, "evaluate": evaluate, "report": report}[args.command](args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error
