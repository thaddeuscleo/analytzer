import gradio as gr
import io

import torch
import torch.nn.functional as F
from torch.utils.model_zoo import load_url
from scipy.special import expit

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils

import librosa
import torchaudio
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead

from matplotlib import pyplot as plt

net_model = "EfficientNetAutoAttB4"
train_db = "DFDC"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
face_policy = "scale"
face_size = 224
frames_per_video = 32


def load_deep_fake_neural_network(net_model, train_db, device):
    model_url = weights.weight_url["{:s}_{:s}".format(net_model, train_db)]
    net = getattr(fornet, net_model)().eval().to(device)
    net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
    return net


def load_face_extractor(device, frames_per_video):
    facedet = BlazeFace().to(device)
    facedet.load_weights("./blazeface/blazeface.pth")
    facedet.load_anchors("./blazeface/anchors.npy")
    videoreader = VideoReader(verbose=False)
    video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)
    return face_extractor


net = load_deep_fake_neural_network(net_model, train_db, device)
transf = utils.get_transformer(
    face_policy, face_size, net.get_normalizer(), train=False
)
face_extractor = load_face_extractor(device, frames_per_video)


def Video_Model(video):
    res, faces_real_pred = predict_deep_fake_video(video)

    chart_filename = create_fakeness_plot(res, faces_real_pred)
    result = "Average score for Fakeness video: {:.4f}".format(
        expit(faces_real_pred.mean())
    )

    return result, chart_filename


def predict_deep_fake_video(video):
    res = face_extractor.process_video(video)
    face_tensor = torch.stack(
        [
            transf(image=frame["faces"][0])["image"]
            for frame in res
            if len(frame["faces"])
        ]
    )
    with torch.no_grad():
        faces_real_pred = net(face_tensor.to(device)).cpu().numpy().flatten()
    return res, faces_real_pred


def create_fakeness_plot(res, faces_real_pred):
    """
    create a plot for showing fakeness confidence on provided frames.
    """
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


def verify_deep_fake_video(video):
    verification_result, filename = Video_Model(video)
    return verification_result, filename


def load_audio(audiopath, sampling_rate=22000):
    audio, lsr = librosa.load(audiopath, sr=sampling_rate)
    audio = torch.FloatTensor(audio)

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with audio data. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)

    return audio.unsqueeze(0)



def classify_audio_clip(clip):
    """
    Returns whether or not the classifier thinks the given clip came from AI generation.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: The probability of the audio clip being AI-generated.
    """
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]


def verify_fake_voice(suspect_sample):
    clip = load_audio(suspect_sample)
    res = classify_audio_clip(clip)
    res = res.item()
    return f"The uploaded audio is {res * 100:.2f} % likely to be AI Generated."


def verify_fake_face(audio):
    return f""


"""
# --------
# |  UI  |
# -------- 
"""

deep_fake_detection_iface = gr.Interface(
    api_name="deep_fake_detection_iface",
    fn=verify_deep_fake_video,
    inputs="video",
    outputs=["text", "image"],
    title="Deep Fake Video Analysis",
    description="Upload a video for deep fake analysis.",
    live=False,
)

with gr.Blocks() as voice_fake_detection_iface:
    gr.Markdown("## Fake Audio Analysis")
    suspect_audio_sample = gr.Audio(label="Suspect Audio Sample", type="filepath");
    text_output = gr.Textbox(label="Audio Fakeness Result")
    check_audio_btn = gr.Button("Submit");
    check_audio_btn.click(verify_fake_voice, inputs=[suspect_audio_sample], outputs=text_output)

fake_face_detection_iface = gr.Interface(
    api_name="fake_face_detection_iface",
    fn=verify_fake_face,
    inputs="image",
    outputs=["text"],
    title="Fake Face Analysis",
    description="Upload a Image for fake audio analysis.",
    live=False,
)

interfaces = [
    deep_fake_detection_iface,
    voice_fake_detection_iface,
    fake_face_detection_iface,
]

tab_names = ["Deep Fake Video Detection", "Fake Voice Detection", "Fake Face Detection"]



app = gr.TabbedInterface(interfaces, tab_names)
app.launch()
