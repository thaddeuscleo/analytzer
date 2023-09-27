import gradio as gr
import io
import torch
from torch.utils.model_zoo import load_url
from scipy.special import expit
from matplotlib import pyplot as plt
from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils
import librosa
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

# Constants
net_model = "EfficientNetAutoAttB4"
train_db = "DFDC"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
face_policy = "scale"
face_size = 224
frames_per_video = 32


# Load deep fake neural network
def load_deep_fake_neural_network(net_model, train_db, device):
    model_url = weights.weight_url[f"{net_model}_{train_db}"]
    net = getattr(fornet, net_model)().eval().to(device)
    net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
    return net


net = load_deep_fake_neural_network(net_model, train_db, device)


# Load face extractor
def load_face_extractor(device, frames_per_video):
    facedet = BlazeFace().to(device)
    facedet.load_weights("./blazeface/blazeface.pth")
    facedet.load_anchors("./blazeface/anchors.npy")
    videoreader = VideoReader(verbose=False)
    video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)
    return face_extractor


face_extractor = load_face_extractor(device, frames_per_video)


# Predict deep fake video
def predict_deep_fake_video(video):
    res = face_extractor.process_video(video)
    face_tensor = torch.stack(
        [
            utils.get_transformer(
                face_policy, face_size, net.get_normalizer(), train=False
            )(image=frame["faces"][0])["image"]
            for frame in res
            if len(frame["faces"])
        ]
    )
    with torch.no_grad():
        faces_real_pred = net(face_tensor.to(device)).cpu().numpy().flatten()
    return res, faces_real_pred


# Create fakeness plot
def create_fakeness_plot(res, faces_real_pred):
    frames = [f["frame_idx"] for f in res if len(f["faces"])]
    fakeness = expit(faces_real_pred)

    plt.figure(figsize=(12, 6))
    plt.bar(frames, fakeness, width=2, color="royalblue")
    plt.xlabel("Frames")
    plt.ylabel("Fakeness (0 to 1)")
    plt.title("Fake Data on Each Frame")
    plt.ylim(0, 1)
    plt.xticks(frames, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    chart_filename = "line_chart.png"
    plt.savefig(chart_filename)
    plt.close()
    return chart_filename


# Verify deep fake video
def verify_deep_fake_video(video):
    res, faces_real_pred = predict_deep_fake_video(video)
    chart_filename = create_fakeness_plot(res, faces_real_pred)
    result = f"Average score for Fakeness video: {expit(faces_real_pred.mean()):.4f}"
    return result, chart_filename


# Extract audio features
def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file)

    one_minute_samples = int(60 * sr)
    y = y[:one_minute_samples]
    segment_duration = sr  # 1 second
    num_segments = len(y) // segment_duration
    chroma_stft_list = []
    rms_list = []
    spectral_centroid_list = []
    spectral_bandwidth_list = []
    rolloff_list = []
    zero_crossing_rate_list = []
    mfcc_matrix = np.zeros((num_segments, 20))

    for i in range(num_segments):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        segment = y[start:end]
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=segment, sr=sr))
        rms = np.mean(librosa.feature.rms(y=segment))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
        spectral_bandwidth = np.mean(
            librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        )
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=segment))
        mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20), axis=1)
        chroma_stft_list.append(chroma_stft)
        rms_list.append(rms)
        spectral_centroid_list.append(spectral_centroid)
        spectral_bandwidth_list.append(spectral_bandwidth)
        rolloff_list.append(rolloff)
        zero_crossing_rate_list.append(zero_crossing_rate)
        mfcc_matrix[i, :] = mfcc

    feature_dict = {
        "Chroma_STFT": chroma_stft_list,
        "RMS": rms_list,
        "Spectral_Centroid": spectral_centroid_list,
        "Spectral_Bandwidth": spectral_bandwidth_list,
        "Spectral_Rolloff": rolloff_list,
        "Zero_Crossing_Rate": zero_crossing_rate_list,
    }
    for i in range(20):
        feature_dict[f"MFCC_{i + 1}"] = mfcc_matrix[:, i]

    df = pd.DataFrame(feature_dict)
    return df


# Verify fake voice
def verify_fake_voice(real_audio_sample, suspect_sample):
    real_df = extract_audio_features(real_audio_sample)
    fake_df = extract_audio_features(suspect_sample)
    real_df["label"] = "REAL"
    fake_df["label"] = "FAKE"
    num_samples_per_class = min(len(real_df), len(fake_df))
    balanced_real_df = real_df.sample(n=num_samples_per_class, random_state=42)
    balanced_fake_df = fake_df.sample(n=num_samples_per_class, random_state=42)
    balanced_all_df = pd.concat([balanced_real_df, balanced_fake_df], axis=0)
    balanced_all_df = balanced_all_df.sample(frac=1, random_state=42).reset_index(
        drop=True
    )
    X = balanced_all_df.iloc[:, :-1]
    y = balanced_all_df.iloc[:, -1]

    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)
    y = y.ravel()

    model = RandomForestClassifier(n_estimators=50, random_state=1)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    acc_score = []
    prec_score = []
    rec_score = []
    f1s = []
    MCCs = []
    ROCareas = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        acc = accuracy_score(pred_values, y_test)
        acc_score.append(acc)
        prec = precision_score(y_test, pred_values, average="binary", pos_label=1)
        prec_score.append(prec)
        rec = recall_score(y_test, pred_values, average="binary", pos_label=1)
        rec_score.append(rec)
        f1 = f1_score(y_test, pred_values, average="binary", pos_label=1)
        f1s.append(f1)
        mcc = matthews_corrcoef(y_test, pred_values)
        MCCs.append(mcc)
        roc = roc_auc_score(y_test, pred_values)
        ROCareas.append(roc)

    accuracy = (
        f"{round(np.mean(acc_score) * 100, 3)}% ({round(np.std(acc_score) * 100, 3)})"
    )
    precision = f"{round(np.mean(prec_score), 3)} ({round(np.std(prec_score), 3)})"
    recall = f"{round(np.mean(rec_score), 3)} ({round(np.std(rec_score), 3)})"
    f1_score_res = f"{round(np.mean(f1s), 3)} ({round(np.std(f1s), 3)})"
    mcc_res = f"{round(np.mean(MCCs), 3)} ({round(np.std(MCCs), 3)})"
    roc_auc = f"{round(np.mean(ROCareas), 3)} ({round(np.std(ROCareas), 3)})"

    global audio_model
    audio_model = model

    return accuracy, precision, recall, f1_score_res, mcc_res, roc_auc


# Analyze Audio using model
def analyze_audio(audio):
    df = extract_audio_features(audio)
    res = audio_model.predict(df)
    return f"{res.mean() * 100} %"


# Fake face detection
def verify_fake_face(audio):
    return ""


with gr.Blocks() as introduction_iface:
    gr.Markdown(
        """
    ## Welcome to Analytzer
    
    In an era where digital manipulation has reached unprecedented levels, 
    discerning the real from the fabricated has become an increasingly challenging task. 
    Enter Analytzer, your comprehensive solution to combat the proliferation of 
    deep fakes and fraudulent content. Analytzer is a cutting-edge application that 
    equips you with the tools to safeguard your digital space.

    With three powerful modules at your disposal, Analytzer empowers you to:
    1. **Deep Fake Video Detection**: Unmask the most sophisticated deep fake videos and safeguard the integrity of visual content. Our state-of-the-art technology scrutinizes every pixel, ensuring that authenticity prevails.
    2. **Fake Voice/Speech Detection**: Hear the truth in every sound. Analytzer's advanced algorithms dissect audio recordings, exposing deceptive voice manipulations and ensuring that genuine voices are heard.
    3. **Deep Fake Face Image Detection**: Protect the visual identity of individuals and organizations. Analytzer's facial recognition prowess pierces through deceptive images, preserving the sanctity of digital profiles.

    In a world where authenticity matters more than ever, Analytzer is your vigilant guardian, enabling you to navigate the digital landscape with confidence. Welcome to the future of content integrity. Welcome to Analytzer.        
    """
    )


# UI Interfaces
with gr.Blocks() as deep_fake_video_detection_iface:
    gr.Markdown(
        "This module is used for detecting video for deep fake using DeepLearning Models."
    )
    with gr.Row():
        with gr.Column():
            suspect_video_input = gr.Video(label="Suspect Video Input")
            analyze_btn = gr.Button(value="Analyze Video", variant="primary")
        with gr.Group():
            fakeness_plot = gr.Image(label="Fakeness On Frames", interactive=True)
            fakeness_result = gr.Text(label="Video Fakeness")
            gr.ClearButton([suspect_video_input, fakeness_result, fakeness_plot])
        analyze_btn.click(
            verify_deep_fake_video,
            inputs=[suspect_video_input],
            outputs=[fakeness_result, fakeness_plot],
        )

with gr.Blocks() as fake_voice_detection_iface:
    gr.Markdown(
        "This module is used for detecting fake speech/ fake voice using ML Models."
    )
    with gr.Accordion("1. Train", open=True):
        gr.Markdown("## Train Fake Audio Model Analysis")
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    real_audio_sample = gr.Audio(
                        label="Real Audio Sample", type="filepath"
                    )
                    fake_audio_sample = gr.Audio(
                        label="Fake Audio Sample", type="filepath"
                    )
                train_audio_btn = gr.Button(value="Train", variant="primary")
            with gr.Group():
                accuracy_output = gr.Textbox(label="Accuracy")
                precision_output = gr.Textbox(label="Precision")
                recall_output = gr.Textbox(label="Recall")
                f1_score_output = gr.Textbox(label="F1-Score")
                mcc_output = gr.Textbox(label="MCC")
                roc_auc_output = gr.Textbox(label="ROC AUC")
                gr.ClearButton(
                    [
                        real_audio_sample,
                        fake_audio_sample,
                        accuracy_output,
                        precision_output,
                        recall_output,
                        f1_score_output,
                        mcc_output,
                        roc_auc_output,
                    ]
                )
            # Event Handling
            train_audio_btn.click(
                verify_fake_voice,
                inputs=[real_audio_sample, fake_audio_sample],
                outputs=[
                    accuracy_output,
                    precision_output,
                    recall_output,
                    f1_score_output,
                    mcc_output,
                    roc_auc_output,
                ],
            )
    with gr.Accordion("2. Analyze", open=False):
        gr.Markdown("## 2. Analyze Audio Using Trained Model")
        with gr.Row():
            with gr.Column():
                suspect_audio_sample = gr.Audio(
                    label="Suspect Audio Sample", type="filepath"
                )
                analyze_audio_btn = gr.Button(value="Analyze", variant="primary")
            with gr.Column():
                fakeness_output = gr.Textbox(label="Realness")
    analyze_audio_btn.click(
        analyze_audio, inputs=[suspect_audio_sample], outputs=[fakeness_output]
    )


with gr.Blocks() as deep_fake_face_image_detection_iface:
    gr.Markdown(
        "This module is used for detecting video for deep fake using DeepLearning Models."
    )
    with gr.Row():
        with gr.Column():
            suspect_image_input = gr.Image(label="Suspect Image Input")
            analyze_btn = gr.Button(value="Analyze Image", variant="primary")
        with gr.Column():
            fakeness_result = gr.Text(label="Image Fakeness")
            gr.ClearButton([suspect_video_input, fakeness_result])
        analyze_btn.click(
            verify_deep_fake_video,
            inputs=[suspect_video_input],
            outputs=[fakeness_result],
        )


interfaces = [
    introduction_iface,
    deep_fake_video_detection_iface,
    fake_voice_detection_iface,
    deep_fake_face_image_detection_iface,
]

tab_names = [
    "* Introduction",
    "A. Deep Fake Video Detection",
    "B. Fake Voice/Speech Detection",
    "C. Deep Fake Face Image Detection",
]

demo = gr.TabbedInterface(
    interfaces,
    tab_names,
    title="🎭 Analytzer",
    css="footer {visibility: hidden}",
)
demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
