import sys
import argparse
import os
import copy
import time
import logging
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--resume', default=None, help='path to the weights to be resumed')
    parser.add_argument(
        '--resume_weights_only',
        action='store_true',
        help='specify this argument to restore only the weights (w/o training states), e.g. --resume path/to/resume --resume_weights_only'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--validate', action='store_true')
    group.add_argument('--test', action='store_true')
    group.add_argument('--predict', action='store_true')
    # group.add_argument('--export', action='store_true') # TODO: a separate export action

    parser.add_argument('--exp_dir', default='./exp')
    parser.add_argument('--runs_dir', default='./runs')
    parser.add_argument('--verbose', action='store_true', help='if true, set logging level to DEBUG')

    args, extras = parser.parse_known_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    import datasets
    import systems
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    from utils.callbacks import CodeSnapshotCallback, ConfigSnapshotCallback, CustomProgressBar
    from utils.misc import load_config    

    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    config.trial_name = config.get('trial_name') or (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    config.exp_dir = config.get('exp_dir') or os.path.join(args.exp_dir, config.name)
    config.save_dir = config.get('save_dir') or os.path.join(config.exp_dir, config.trial_name, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')
    config.config_dir = config.get('config_dir') or os.path.join(config.exp_dir, config.trial_name, 'config')

    logger = logging.getLogger('pytorch_lightning')
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if 'seed' not in config:
        config.seed = int(time.time() * 1000) % 1000
    pl.seed_everything(config.seed)

    dm = datasets.make(config.dataset.name, config.dataset)
    system = systems.make(config.system.name, config, load_from_checkpoint=None if not args.resume_weights_only else args.resume)

    callbacks = []
    if args.train:
        
        folders_to_ignore = [
            'data/',
            'output/',
            'runs/', # Essential to ignore the TensorBoard log directory
            'exp/',  # Essential to ignore the experiment directory where snapshots are saved
            '__pycache__/',
            '*.log',
            '.git/', # Although git ls-files often excludes this by default, explicit is good.
            'venv/', # If you have a virtual environment folder
            'env/',  # Another common virtual environment name
            'wandb/', # If you use Weights & Biases
        ]

        callbacks += [
            ModelCheckpoint(
                dirpath=config.ckpt_dir,
                **config.checkpoint
            ),
            LearningRateMonitor(logging_interval='step'),
            CodeSnapshotCallback(
                config.code_dir,
                use_version=False, # As per your existing setup
                ignore_patterns=folders_to_ignore # Pass the list of patterns here
            ),
            ConfigSnapshotCallback(
                config, config.config_dir, use_version=False
            ),
            CustomProgressBar(refresh_rate=1),
        ]

    loggers = []
    if args.train:
        loggers += [
            TensorBoardLogger(args.runs_dir, name=config.name, version=config.trial_name),
            CSVLogger(config.exp_dir, name=config.trial_name, version='csv_logs')
        ]
    
    if n_gpus == 1:
        strategy = 'auto'
    elif sys.platform == 'win32':
        strategy = 'dp'
    else:
        strategy = 'ddp_find_unused_parameters_false'
    
    trainer = Trainer(
        devices=n_gpus,
        accelerator='gpu',
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        **config.trainer
    )

    if args.train:
        from pytorch_lightning.utilities.rank_zero import rank_zero_info

        albedo_cfg = config.system.get('albedo_scaling', {})
        two_phase_explicit = albedo_cfg.get('enabled', None)

        # Auto-detect: enable two-phase when albedos are present (unless explicitly disabled)
        if two_phase_explicit is None:
            has_albedos = False
            ds = config.dataset
            if ds.name == 'sfm' and ds.get('albedo_sfm', ''):
                has_albedos = True
            elif ds.name == 'idr' and os.path.isdir(os.path.join(ds.root_dir, 'albedo')):
                has_albedos = True
            two_phase = has_albedos
        else:
            two_phase = two_phase_explicit

        if two_phase:
            from utils.albedo_scaling import compute_albedo_scale_ratios, scale_albedo_images
            import numpy as np

            total_steps = config.trainer.max_steps
            warmup_ratio = albedo_cfg.get('warmup_ratio', 0.1)
            phase1_steps = int(warmup_ratio * total_steps)

            # Validate: rendering lambdas must be scalar (not schedules)
            for key in ['lambda_rendering_mse', 'lambda_rendering_l1']:
                val = config.system.loss[key]
                if isinstance(val, (list, tuple)):
                    raise ValueError(f"Two-phase training requires scalar {key}, got schedule: {val}")

            # Helper: recompute scheduler gamma for a given max_steps
            warmup_steps = config.system.warmup_steps
            def _recompute_scheduler(cfg, max_steps):
                cfg.trainer.max_steps = max_steps
                decay_steps = max(max_steps - warmup_steps, 1)
                new_gamma = 0.1 ** (1.0 / decay_steps)
                # Update the ExponentialLR gamma (second scheduler in the list)
                cfg.system.scheduler.schedulers[1].args.gamma = new_gamma
                cfg.checkpoint.every_n_train_steps = max_steps

            # ---- PHASE 1: geometry only (no_albedo, rendering loss stays active) ----
            rank_zero_info(f"[TwoPhase] Phase 1: {phase1_steps} steps, no_albedo=True (shading-only rendering)")
            config_p1 = copy.deepcopy(config)
            _recompute_scheduler(config_p1, phase1_steps)
            config_p1.model.no_albedo = True  # Freeze texture, force rgb=1

            system_p1 = systems.make(
                config_p1.system.name, config_p1,
                load_from_checkpoint=None if not args.resume_weights_only else args.resume
            )
            dm.setup('fit')  # Explicit setup before phase 1

            # Force GT albedos to white for phase 1 (rendering = shading * 1.0)
            ds = dm.train_dataloader().dataset
            import torch as _torch
            real_albedos = ds.all_images.clone()
            ds.all_images = _torch.ones_like(ds.all_images)

            # Phase-specific callbacks (avoid stale state from shared callbacks)
            callbacks_p1 = [
                ModelCheckpoint(dirpath=config_p1.ckpt_dir, **config_p1.checkpoint),
                LearningRateMonitor(logging_interval='step'),
                CustomProgressBar(refresh_rate=1),
            ]
            trainer_p1 = Trainer(
                devices=n_gpus, accelerator='gpu',
                callbacks=callbacks_p1, logger=loggers, strategy=strategy,
                **config_p1.trainer
            )

            if args.resume and not args.resume_weights_only:
                trainer_p1.fit(system_p1, datamodule=dm, ckpt_path=args.resume)
            else:
                trainer_p1.fit(system_p1, datamodule=dm)

            # Restore real albedos (needed for albedo scaling)
            ds.all_images = real_albedos

            # ---- ALBEDO SCALING: extract mesh + compute ratios ----
            rank_zero_info("[TwoPhase] Extracting intermediate mesh for albedo scaling")
            system_p1.model.cuda()  # Ensure model is on GPU after trainer.fit()
            mesh_res = 512  # Fixed resolution for intermediate mesh
            # Recreate marching-cubes helper at desired resolution
            from models.geometry import MarchingCubeHelper
            geom = system_p1.model.geometry
            use_torch = config.model.geometry.isosurface.method == 'mc-torch'
            geom.helper = MarchingCubeHelper(mesh_res, use_torch=use_torch)

            # Export WITHOUT vertex colors (useless in phase 1 with no_albedo).
            # Export in normalized space: verts are in the same frame as all_c2w cameras
            # (no inverse-scaling needed for albedo scaling / P2 renorm).
            from copy import deepcopy
            from datasets.utils import compute_scaling_from_mesh
            export_cfg_p1 = deepcopy(config.export)
            export_cfg_p1.export_vertex_color = False
            export_cfg_p1.isosurface_space = 'normalized'
            mesh = system_p1.model.export(export_cfg_p1)

            import trimesh
            verts_norm = mesh['v_pos'].cpu().numpy()  # P1-normalized space
            faces = mesh['t_pos_idx'].cpu().numpy()
            del mesh  # Free tensor memory

            # Save intermediate mesh in WORLD space (for debug tools / external viewers).
            # Inverse-transform: v_world = v_norm / p1_scale + p1_center
            p1_center = np.array(ds.scene_center, dtype=np.float64)
            p1_scale  = float(ds.scale_factor)
            verts_world_inter = verts_norm / p1_scale + p1_center
            os.makedirs(config.save_dir, exist_ok=True)
            trimesh.Trimesh(vertices=verts_world_inter, faces=faces).export(
                os.path.join(config.save_dir, 'intermediate_mesh.ply')
            )

            # ---- SCENE RENORMALIZATION from intermediate mesh ----
            # sphere_scale = config.dataset.get('sphere_scale', 0.9)  # old: used phase-1 scale
            sphere_scale_p2 = 1.5  # Fill model sphere (radius=1.5) to minimize floater zones
            new_center, new_scale = compute_scaling_from_mesh(verts_world_inter, sphere_scale=sphere_scale_p2)
            ds.scene_center = new_center
            ds.scale_factor = new_scale
            rank_zero_info(f"[TwoPhase] Scene renormalized: center={new_center.tolist()}, scale={new_scale:.6f}")

            # Free GPU memory BEFORE CPU-heavy albedo scaling (ray tracing)
            del system_p1, trainer_p1
            import gc
            gc.collect()
            _torch.cuda.empty_cache()

            # Build mesh in normalized space (consistent with all_c2w cameras)
            tri_mesh_norm = trimesh.Trimesh(vertices=verts_norm, faces=faces)

            # Build normalized cameras from all_c2w (undo NeuS Y/Z flip)
            all_c2w_np = ds.all_c2w.cpu().numpy()  # (N, 3, 4), normalized space, Y/Z flipped
            norm_c2ws = []
            for c2w34 in all_c2w_np:
                c2w44 = np.eye(4, dtype=np.float64)
                c2w44[:3, :4] = c2w34
                c2w44[:3, 1:3] *= -1.  # undo NeuS Y/Z flip → standard convention
                norm_c2ws.append(c2w44)

            scale_ratios = compute_albedo_scale_ratios(
                albedo_images=[img.cpu().numpy() for img in ds.all_images],
                camera_Ks=ds.camera_Ks,
                camera_c2ws=norm_c2ws,
                tri_mesh=tri_mesh_norm,
                n_samples=albedo_cfg.get('n_samples', 2000),
            )
            del tri_mesh_norm  # Free ray tracer BVH
            scaled = scale_albedo_images(ds.all_images, scale_ratios)
            ds.update_albedos(scaled)
            rank_zero_info(f"[TwoPhase] Albedos scaled. Ratios per view: {scale_ratios.tolist()}")
            rank_zero_info(f"[TwoPhase] Mean={scale_ratios.mean(axis=0).tolist()}, "
                           f"Min={scale_ratios.min(axis=0).tolist()}, Max={scale_ratios.max(axis=0).tolist()}")

            # ---- PHASE 2: fresh model, full training with scaled albedos ----
            rank_zero_info(f"[TwoPhase] Phase 2: {total_steps} steps, fresh model, rendering loss active")
            config_p2 = copy.deepcopy(config)
            _recompute_scheduler(config_p2, total_steps)

            system_p2 = systems.make(config_p2.system.name, config_p2)  # Fresh weights
            callbacks_p2 = [
                ModelCheckpoint(dirpath=config_p2.ckpt_dir, **config_p2.checkpoint),
                LearningRateMonitor(logging_interval='step'),
                CodeSnapshotCallback(config_p2.code_dir, use_version=False, ignore_patterns=folders_to_ignore),
                ConfigSnapshotCallback(config_p2, config_p2.config_dir, use_version=False),
                CustomProgressBar(refresh_rate=1),
            ]
            trainer_p2 = Trainer(
                devices=n_gpus, accelerator='gpu',
                callbacks=callbacks_p2, logger=loggers, strategy=strategy,
                **config_p2.trainer
            )

            # dm.setup() is IDEMPOTENT — won't overwrite scaled albedos
            trainer_p2.fit(system_p2, datamodule=dm)
            trainer_p2.test(system_p2, datamodule=dm)

        else:
            # Standard single-phase training (unchanged)
            if args.resume and not args.resume_weights_only:
                trainer.fit(system, datamodule=dm, ckpt_path=args.resume)
            else:
                trainer.fit(system, datamodule=dm)
            trainer.test(system, datamodule=dm)

    elif args.validate:
        trainer.validate(system, datamodule=dm, ckpt_path=args.resume)
    elif args.test:
        trainer.test(system, datamodule=dm, ckpt_path=args.resume)
    elif args.predict:
        trainer.predict(system, datamodule=dm, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
