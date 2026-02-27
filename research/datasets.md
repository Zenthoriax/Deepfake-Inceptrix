# Training Datasets Reference

This document outlines the major datasets required for training the deepfake detection models, their contents, and access methods.

## ⚠️ Important License Notice
All major datasets listed below are for **research use only**. They require agreement acceptance, cannot be redistributed, and cannot be used commercially without explicit permission.

---

## 1. FaceForensics++
*   **Contents:** Real videos, DeepFakes, FaceSwap, Face2Face, NeuralTextures (multiple compression levels).
*   **Size:** ~500GB+
*   **Best For:** Baseline model training (e.g., EfficientNet-B4), Academic benchmarking.
*   **Access:** Official website research access form. Provides download scripts.

## 2. DeepFake Detection Challenge (DFDC)
*   **Contents:** 100,000+ videos. Highly diverse actors, real-world style deepfakes.
*   **Size:** ~470GB compressed.
*   **Best For:** Large-scale training, Generalization.
*   **Access:** Kaggle competition rules acceptance. `kaggle competitions download -c deepfake-detection-challenge`

## 3. Celeb-DF
*   **Contents:** More realistic deepfakes, High-quality manipulations.
*   **Size:** ~300GB.
*   **Best For:** Testing model robustness, Cross-dataset evaluation.
*   **Access:** Official Celeb-DF website. Email-based access approval.

## 4. FakeAVCeleb (Audio + Video)
*   **Contents:** Face swaps, Voice cloning, Lip-sync deepfakes.
*   **Best For:** Multimodal training, Audio-visual consistency modeling (e.g., audio/video fusion pipelines).
*   **Access:** Official research site request form.

## 5. ASVspoof (Audio Only)
*   **Contents:** Voice spoofing samples, Synthetic speech, Replay attacks.
*   **Best For:** Training audio models (Wav2Vec 2.0, ECAPA-TDNN, RawNet2).
*   **Access:** Official ASVspoof website research registration.

## 6. Google Deepfake Detection Dataset
*   **Contents:** Released by Google for DFDC.
*   **Best For:** Expanding the DFDC training corpus.
*   **Access:** Provided alongside DFDC / Google datasets portal.
