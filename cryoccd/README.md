# CryoCCD

CryoCCD is a cycle-consistent diffusion framework for unpaired translation between synthetic and real Cryo-EM micrographs, featuring mask-guided contrastive learning, GAN supervision, and optional window attention.

---

## 1. Change to the project directory

```bash
cd cryoccd
```

---

## 2. Install requirements

Make sure your environment supports PyTorch with GPU and CUDA.

```bash
pip install -r requirements.txt
```

---

## 3. Training

Run the following command to train the CryoCCD model:

```bash
python -m cryoccd.commands.train \
  --model cryoccd \
  --name <experiment_name> \
  --max_dataset_size 300 \
  --apix 5.36 \
  --real_dir <path_to_real_images> \
  --sync_dir <path_to_synthetic_images> \
  --mask_dir <path_to_masks> \
  --weight_map_dir <path_to_weight_maps> \
  --T 10 \
  --beta_1 0.0001 \
  --beta_T 0.02 \
  --beta1 0.5 \
  --lr 1e-4 \
  --sampling_steps 10 \
  --lambda_NCE 20.0 \
  --lambda_GAN 1.0 \
  --lambda_cycle 5.0 \
  --use_window_attention True \
  --window_size 8 \
  --num_heads 4 \
  --sampler_type ddpm
```

> Replace `<...>` with your actual paths and experiment name.

---

## 4. Testing / Sampling

Run the following command to sample/test from a trained model:

```bash
python -m cryoccd.commands.test \
  --name <experiment_name> \
  --max_dataset_size 1000 \
  --num_test 1000 \
  --apix 5.36 \
  --sync_dir <path_to_synthetic_images> \
  --mask_dir <path_to_masks> \
  --pose_dir <path_to_pose_files> \
  --weight_map_dir <path_to_weight_maps> \
  --save_dir <output_directory> \
  --sampling_steps 10 \
  --sampler_type ddpm \
  --T 10
```

---

## 5. Sampler types

CryoCCD supports the following samplers (choose with `--sampler_type`):

- `ddpm`
- `ddim`
- `dpmsolver`
- `dpmsolver++`

---

## Notes

- `--apix` is the Ångström-per-pixel value (e.g., 5.36 for Ribosome).
- `--sampling_steps` controls the number of denoising steps at inference.
- `--T` specifies the number of total diffusion steps during training and must match during testing.
- `--use_window_attention` enables localized attention in the U-Net.
- All image directories should contain identically named PNG or MRC files.

---
