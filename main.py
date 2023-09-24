import gradio as gr

import torch
from torch.utils.model_zoo import load_url
from scipy.special import expit

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils

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


def verify_fake_voice(suspect_sample, calibration_sample):
    return f""


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
    suspect_audio_sample = gr.Audio(label="Suspect Audio Sample");
    calibration_audio_sample = gr.Audio(label="Calibration Audio Sample");
    text_output = gr.Textbox(label="Audio Fakeness Result")
    check_audio_btn = gr.Button("Submit");
    check_audio_btn.click(verify_fake_voice, inputs=[suspect_audio_sample, calibration_audio_sample], outputs=text_output)

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
