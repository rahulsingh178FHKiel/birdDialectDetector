# Dataset Characteristics

## Dataset Information

### Dataset Source
* **Dataset Link:** [Xeno-Canto Open-Access Database](https://xeno-canto.org/)
* **Dataset Owner/Contact:** Xeno-Canto Foundation (Open-access community collection)

### Dataset Characteristics
* **Number of Observations:** 155 high-quality bird vocalizations in the primary metadata index (which is filtered dynamically based on classification mode).
* **Number of Features:** 
  * **Metadata Features:** 16 metadata columns parsed from the Xeno-Canto API.
  * **Model Input Features:** 2D Log-Mel spectrogram array of shape `(1, 128, 258)` representing a resampled (22,050 Hz) mono audio clip cropped to a fixed 3.0-second window.

### Target Variable/Label
* **Label Name:** `target_region` / `binary_label`
* **Label Type:** Classification (Multi-class regional classification or binary island-vs-continent classification).
* **Label Description:** The goal is to predict the geographic origin (regional dialect classification) of a Yellowhammer (*Emberiza citrinella*) recording based on its vocal signatures.
* **Label Values:**
  * **Binary Mode:** `continental` (0) vs. `island` (1 - southern_uk).
  * **5-Class Mode:** `east_germany` (0), `netherlands_belgium` (1), `south_poland` (2), `southern_sweden` (3), `southern_uk` (4).
* **Label Distribution (Unfiltered):**
  * `netherlands_belgium`: 50 samples
  * `east_germany`: 31 samples
  * `southern_sweden`: 31 samples
  * `south_poland`: 22 samples
  * `southern_uk`: 20/92 samples (filtered down to 20 for balanced training trials)

---

## Feature Description

### Metadata features (parsed from CSV)
* **id (`int`):** Unique Xeno-Canto recording ID.
* **rec (`string`):** Name of the recordist. Used for deduplication to prevent model learning recordist-specific microphone signatures.
* **cnt (`string`):** Country where the recording was registered.
* **lat / lon (`float`):** GPS coordinates. Used to assign recordings to strict, non-overlapping regional bounding boxes.
* **q (`string`):** Quality grade (A or B only) to ensure clear signal-to-noise ratio.
* **length (`string`):** Duration of original file.
* **target_region (`string`):** Assigned geographic label class based on regional bounding boxes.

### Engineered Model Features (generated in dataset pipeline)
* **Log-Mel Spectrogram (`float32` array):** Log-scaled Mel-spectrogram representing audio energy distribution across time and frequency. Time resolution is determined by a hop length of 256 samples (22.05 kHz sample rate).
* **Cropping & Denoising:** Automatically cropped to a 3-second window ending at the song's final active frame in the 2-9 kHz band. Broad-band background noise is suppressed using median row-subtraction.

---

## Exploratory Data Analysis

The exploratory data analysis is conducted within the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which performs:
* **Data loading and initial inspection** of geographic distributions and metadata attributes.
* **Visualizing input spectrograms** to verify the quality of the 3-second dialect window crop.
* **Class distribution summaries** dynamically matching the current selected classification mode (3-class, 4-class, 5-class, or binary).
* **Dynamic baseline computation** mapping exact random chance thresholds based on current target class splits.
