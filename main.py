import gradio as gr

import torch
from torch.utils.model_zoo import load_url
from scipy.special import expit

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet,weights
from isplutils import utils

from matplotlib import pyplot as plt

net_model = 'EfficientNetAutoAttB4'
train_db = 'DFDC'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 32

model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
net = getattr(fornet,net_model)().eval().to(device)
net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))

transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

facedet = BlazeFace().to(device)
facedet.load_weights("./blazeface/blazeface.pth")
facedet.load_anchors("./blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=facedet)


def Video_Model(video):
    res = face_extractor.process_video(video)
    face_tensor = torch.stack( [ transf(image=frame['faces'][0])['image'] for frame in res if len(frame['faces'])] )
    with torch.no_grad():
        faces_real_pred = net(face_tensor.to(device)).cpu().numpy().flatten()
    
    frames = [f['frame_idx'] for f in res if len(f['faces'])]
    fakeness = expit(faces_real_pred)

    plt.figure(figsize=(12, 6))
    plt.bar(frames, fakeness, width=2, color='royalblue')
    plt.xlabel('Frames')
    plt.ylabel('Fakeness (0 to 1)')
    plt.title('Fake Data on Each Frame')
    plt.ylim(0, 1)
    plt.xticks(frames, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    chart_filename = "line_chart.png"
    plt.savefig(chart_filename)
    plt.close()

    return 'Average score for Fakeness video: {:.4f}'.format(expit(faces_real_pred.mean())), chart_filename


def verify_deep_fake_video(video):
    res = Video_Model(video)
    verification_result, filename = res
    return verification_result, filename
    

iface = gr.Interface(
    fn=verify_deep_fake_video,
    inputs="video",
    outputs=["text", "image"],
    title="Deep Fake Video Verification",
    description="Upload a video for deep fake verification.",
    capture_session=True,  # Allows capturing the video for processing
    live=False
)


iface.launch()