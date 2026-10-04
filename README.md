# CryoCCD

**Simulating Cryo-EM: Cycle-Consistent Predictor–Corrector Diffusion with Biophysical Modeling**
Transactions on Machine Learning Research (TMLR), 2026 · [OpenReview](https://openreview.net/forum?id=oBBBg2MbGB) · [Project page](https://opensciagent.github.io/CryoCCD/)

## Project Structure

This repository is organized into two main modules, plus the project page in `docs/`:

---

### 📁 `cryoem_simulator/`

This folder contains the **data generation pipeline** used to simulate Cryo-EM micrographs

→ **Please refer to this folder for all data construction details.**

---

### 📁 `cryoccd/`

This folder implements our **CryoCCD model**, including:
- The diffusion-based training and sampling code,
- Mask-guided contrastive loss,
- Sampler integration (DDPM, DDIM etc.),
- GAN loss, cycle loss, and window attention.

→ **Please explore this folder for algorithmic and training details.**

---

All EMPIAR datasets can be downloaded in https://www.ebi.ac.uk/empiar/


---

### 📁 `docs/`

Static project page (`index.html`, figures, paper PDF). To publish it: make the repository public, then
**Settings → Pages → Deploy from a branch → `main` / `/docs`**. It is served at
`https://<owner>.github.io/CryoCCD/`; if the owner or repository name changes, update the `canonical`
and `og:image` URLs and the Code button in `docs/index.html`.
