import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual_encoders.dlp.train_dlp import train_dlp


def resolve_run_prefix(config):
    if config.get('run_prefix'):
        return config['run_prefix']
    return f"_{config['n_kp_enc']}kp_{config['n_kp_prior']}kpp_{config['learned_feature_dim']}zdim"


def find_run_dir(log_root, before_dirs):
    if not log_root.exists():
        return None

    after_dirs = {path.resolve() for path in log_root.iterdir() if path.is_dir()}
    new_dirs = sorted(after_dirs - before_dirs, key=lambda path: path.stat().st_mtime)
    if new_dirs:
        return new_dirs[-1]

    if after_dirs:
        return sorted(after_dirs, key=lambda path: path.stat().st_mtime)[-1]

    return None


def export_checkpoint(run_dir, ds_name, run_prefix, export_dir):
    save_dir = run_dir / 'saves'
    candidates = [
        save_dir / f'{ds_name}_dlp{run_prefix}_best.pth',
        save_dir / f'{ds_name}_dlp{run_prefix}.pth',
    ]
    checkpoint_path = next((path for path in candidates if path.exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(f'Could not find a trained DLP checkpoint under {save_dir}.')

    hparams_path = run_dir / 'hparams.json'
    if not hparams_path.exists():
        raise FileNotFoundError(f'Could not find hparams.json under {run_dir}.')

    export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hparams_path, export_dir / 'hparams.json')
    shutil.copy2(checkpoint_path, export_dir / 'best.pth')

    print(f'Exported DLP checkpoint to: {export_dir}')
    print(f'  hparams: {export_dir / "hparams.json"}')
    print(f'  weights: {export_dir / "best.pth"}')


def main():
    parser = argparse.ArgumentParser(description='Train a DLP encoder from a YAML config and export it to chkpts/.')
    parser.add_argument('--config', required=True, help='Path to a YAML config file.')
    parser.add_argument('--export-dir', default=None, help='Optional export directory for best.pth + hparams.json.')
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    run_prefix = resolve_run_prefix(config)

    log_root = (PROJECT_ROOT / 'visual_encoders/dlp/logs').resolve()
    before_dirs = {path.resolve() for path in log_root.iterdir() if path.is_dir()} if log_root.exists() else set()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dlp(
        ds=config['ds'],
        tasks=config.get('tasks'),
        data_root_dir=config['data_root_dir'],
        run_prefix=run_prefix,
        device=device,
        load_model=config['load_model'],
        lr=config['lr'],
        batch_size=config['batch_size'],
        weight_decay=config['weight_decay'],
        dropout=config['dropout'],
        num_epochs=config['num_epochs'],
        warmup_epoch=config['warmup_epoch'],
        kp_activation=config['kp_activation'],
        kp_range=config['kp_range'],
        n_kp=config['n_kp'],
        n_kp_enc=config['n_kp_enc'],
        n_kp_prior=config['n_kp_prior'],
        pad_mode=config['pad_mode'],
        sigma=config['sigma'],
        patch_size=config['patch_size'],
        beta_kl=config['beta_kl'],
        beta_rec=config['beta_rec'],
        kl_balance=config['kl_balance'],
        learned_feature_dim=config['learned_feature_dim'],
        bg_learned_feature_dim=config['bg_learned_feature_dim'],
        recon_loss_type=config['recon_loss_type'],
        topk=config['topk'],
        anchor_s=config['anchor_s'],
        scale_std=config['scale_std'],
        offset_std=config['offset_std'],
        eval_epoch_freq=config['eval_epoch_freq'],
        eval_im_metrics=config['eval_im_metrics'],
        num_workers=config.get('num_workers', 4),
    )

    if args.export_dir:
        run_dir = find_run_dir(log_root, before_dirs)
        if run_dir is None:
            raise RuntimeError('Training finished, but no DLP log directory could be located.')
        export_checkpoint(run_dir, config['ds'], run_prefix, Path(args.export_dir).expanduser().resolve())


if __name__ == '__main__':
    main()
