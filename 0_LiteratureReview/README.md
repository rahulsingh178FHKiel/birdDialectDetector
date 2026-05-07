# Literature Review

Approaches or solutions that have been tried before on similar projects.

**Summary of Each Work**:

- **Source 1**: Identification of dialects and individuals of globally
threatened Yellow Cardinals using neural networks

  - **[Link](https://www.biorxiv.org/content/10.1101/2023.06.07.544140v1)**
  - **Objective**: To use deep learning to identify regional dialects and individual vocal signatures of the endangered Yellow Cardinal from noisy audio field recordings, aiming to assist in non-invasive monitoring and conservation.
  - **Methods**:The researchers converted audio recordings into log-Mel spectrograms and trained Convolutional Neural Networks (CNNs). They utilized bioacoustic-friendly data augmentation techniques, specifically "time masking" and "random cropping," to artificially increase their training data without destroying the biological meaning of the bird songs.
  - **Outcomes**:The CNN successfully classified the bird's geographic dialect with 84% accuracy. Crucially, the model proved that Yellow Cardinal vocalizations have a hierarchical structure, where the acoustic differences between regions (dialects) are significantly larger than the differences between individual birds.
  - **Relation to the Project**: The paper provides the blueprint of the CNN and in general the whole methodology (like the Digital Signal Processing pipeline to convert audio recordings to usable spectrograms/images) that is needed to execute the project. 

- **Source 2**: Bird Sound Classification using Deep Neural Networks: A Comparative Analysis of State-of-the-Art Models

  - **[Link](https://ijisae.org/index.php/IJISAE/article/view/3596)**
  - **Objective**: To evaluate the effectiveness of deep neural network (DNN) architectures for automatic bird sound classification across multiple species.
  - **Methods**: The researchers used datasets such as Xeno-Canto and BirdCLEF and converted raw audio into Mel spectrograms and extracted MFCC features. They also applied supervised learning with labeled bird species data.
  - **Outcomes**: The deep CNN architectures achieved strong classification accuracy and spectrogram-based representations significantly improved performance over raw audio
  - **Relation to the Project**: This work directly supports our pipeline design (spectrogram generation + CNN models). The same approach can be adapted to regional/dialect classification, since dialect differences also manifest as spectro-temporal patterns.

- **Source 3**: Multi-Label Bird Species Classification Using Sequential Aggregation Strategy from Audio Recordings

  - **[Link](https://www.cai.sk/ojs/index.php/cai/article/view/2023_5_1255)**
  - **Objective**: To classify multiple bird species present in a single audio recording using deep learning.
  - **Methods**: The researchers combined CNN (DCNN) and RNN architectures and extracted Mel spectrograms and MFCC features. They also introduced a sequential aggregation strategy to improve predictions across audio segments.
  - **Outcomes**: They achieved strong performance (F1 score up to 0.75) and demonstrated that combining CNN + temporal models improves classification. Showed robustness in multi-label, noisy environments
  - **Relation to the Project**: Our project can benefit from this hybrid approach because dialect detection may depend on temporal patterns in vocalizations. Using CNNs for spatial features (spectrograms) and RNNs/transformers for temporal variation could improve regional prediction.
