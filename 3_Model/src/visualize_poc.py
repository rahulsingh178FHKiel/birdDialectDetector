import os
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

# DATA_DIR = os.path.join("..", "data")
# METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")

DATA_DIR = "/content/drive/MyDrive/Germany/Studies/AppliedML/data"
METADATA_FILE = "/content/drive/MyDrive/Germany/Studies/AppliedML/data/metadata.csv"


def plot_spectrogram(spec, title, ax):
    """Plot the spectrogram on the provided matplotlib axis."""
    im = ax.imshow(spec.numpy(), origin='lower', aspect='auto', cmap='magma')
    ax.set_title(title)
    ax.set_ylabel('Mel bins (2kHz - 9kHz)')
    ax.set_xlabel('Frames')

def crop_end_of_song(waveform, sample_rate, duration=3.0):
    """
    Finds the absolute last moment of bird activity in the 2-9kHz band
    and crops the 3-second window ending exactly at that moment.
    """
    # 1. Quick Spectrogram to find energy in the Yellowhammer band
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=512, f_min=2000.0, f_max=9000.0
    )
    mel_spec = mel_transform(waveform)
    
    # 2. Calculate rolling energy envelope
    energy_envelope = torch.sum(mel_spec, dim=1).squeeze(0)
    
    # 3. Dynamic Threshold: Mean + 0.5 * StdDev
    threshold = energy_envelope.mean() + (0.5 * energy_envelope.std())
    active_frames = torch.where(energy_envelope > threshold)[0]
    
    samples_to_keep = int(duration * sample_rate)
    
    if len(active_frames) == 0:
        # Fallback if too quiet
        if waveform.shape[1] > samples_to_keep:
            return waveform[:, -samples_to_keep:]
        return waveform
        
    # 4. Find the LAST active moment (end of the dialect flourish)
    last_active_frame = active_frames[-1].item()
    last_active_sample = last_active_frame * 512
    
    # 5. Crop 3-second window ending right after the last activity
    padding = int(0.2 * sample_rate) # 200ms padding so we don't clip the tail
    end_sample = min(waveform.shape[1], last_active_sample + padding)
    start_sample = max(0, end_sample - samples_to_keep)
    
    return waveform[:, start_sample:end_sample]

def main():
    if not os.path.exists(METADATA_FILE):
        print("Metadata file not found. Run download_data.py first.")
        return

    df = pd.read_csv(METADATA_FILE)
    if len(df) == 0:
        print("No recordings found in metadata.")
        return

    # Extract unique regions for plotting rows
    regions = df['target_region'].unique()
    
    # Create a grid (Regions as rows, 5 recordings as columns)
    fig, axes = plt.subplots(len(regions), 5, figsize=(20, 3 * len(regions)))
    if len(regions) == 1:
        axes = [axes]

    for row_idx, region in enumerate(regions):
        region_df = df[df['target_region'] == region].head(5)
        
        for col_idx, (_, rec) in enumerate(region_df.iterrows()):
            file_path = os.path.join(DATA_DIR, region, f"{rec['id']}.mp3")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
                
            # --- 1. Audio Loading & Preprocessing ---
            waveform, sample_rate = torchaudio.load(file_path)
            if sample_rate != 22050:
                resampler = T.Resample(sample_rate, 22050)
                waveform = resampler(waveform)
                sample_rate = 22050
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # --- 2. Advanced End-Of-Song Cropping ---
            waveform = crop_end_of_song(waveform, sample_rate, duration=3.0)
                
            # --- 3. Mel-Spectrogram Generation ---
            mel_spectrogram = T.MelSpectrogram(
                sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=128, f_min=2000.0, f_max=9000.0
            )
            mel_spec = mel_spectrogram(waveform)
            log_mel_spec = T.AmplitudeToDB()(mel_spec)[0]
            
            # --- 4. Spectral Noise Reduction (Dampening Wind/Rain) ---
            # Subtract the median energy of each frequency band over time
            median_noise = torch.median(log_mel_spec, dim=1, keepdim=True)[0]
            log_mel_spec = log_mel_spec - median_noise
            log_mel_spec = torch.clamp(log_mel_spec, min=0) # Remove negative artifacts
            
            # --- 5. Plotting ---
            ax = axes[row_idx][col_idx] if len(regions) > 1 else axes[col_idx]
            plot_spectrogram(log_mel_spec, f"{region.replace('_', ' ').upper()} - XC{rec['id']}", ax)

    plt.tight_layout()
    output_path = "poc_spectrograms.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSuccessfully saved visualization to: {output_path}")

if __name__ == '__main__':
    main()
