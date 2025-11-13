# app_camera_input.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from torch import nn
import timm
import gdown
import os
import io
from PIL import Image
import numpy as np
import time
import base64
import tempfile

st.set_page_config(page_title="Violence Detection (Browser Camera)", layout="wide")

st.markdown("""
# 🎥 Violence Detection (Browser Camera)
شغّل الكاميرا من المتصفح (موبايل/لابتوب)، التطبّيقات تستخدم ViT + LSTM لتحليل 8 فريمات متتابعة.
عند اكتشاف سلوك عنيف سيظهر تنبيه بصري ومحاولة تشغيل صوت الإنذار (قد يتطلب سماح التبويب لتشغيل الصوت).
""")

# ---------- تعديل الرابط هنا إلى رابط Google Drive 'uc?id=...' ----------
MODEL_PATH = "best_vit_lstm.pt"
MODEL_DRIVE_ID = "1GjmrQSLRtCwAtkk30ZOtFFXFqhOg6BxX"   # ضع id من رابطك
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
ALERT_AUDIO = "alert.wav"   # ضع alert.wav في نفس المجلد بالمشروع أو في repo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download model if missing
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded.")

# ----------------- نموذج (مطابق للي عندك) -----------------
class ViT_LSTM_Classifier(nn.Module):
    def __init__(self, vit_name="vit_tiny_patch16_224", lstm_hidden=256, lstm_layers=1, num_classes=2, dropout=0.3):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=False, num_classes=0)
        self.feat_dim = self.vit.num_features if hasattr(self.vit, "num_features") else 192
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.vit(x)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits

# load model
model = ViT_LSTM_Classifier().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

st.sidebar.header("Control")
seq_len = st.sidebar.slider("Sequence length (frames)", min_value=4, max_value=16, value=8, step=1)
start = st.sidebar.button("Start stream (use camera button below)")

st.info("اضغط زر الكاميرا أدناه لفتح الكاميرا في المتصفح ثم التقط صورًا متتابعة (أو ادي للكاميرا تحديث متكرر).")

frames_buffer = []

col1, col2 = st.columns([2,1])
with col1:
    cam_file = st.camera_input("Open camera (mobile / desktop)", key="cam")
    if cam_file is not None:
        # convert to numpy array
        img = Image.open(cam_file).convert("RGB")
        img_np = np.array(img)[:,:,::-1]  # BGR for OpenCV-style
        # stream processing: append transformed frame
        tensor_frame = transform(img_np)
        frames_buffer.append(tensor_frame)
        if len(frames_buffer) > seq_len:
            frames_buffer.pop(0)

        # show preview
        st.image(img, caption="Captured frame", use_column_width=True)

with col2:
    st.markdown("### Status")
    st.write(f"Buffered frames: {len(frames_buffer)}/{seq_len}")

    # Play alarm HTML element
    if os.path.exists(ALERT_AUDIO):
        audio_bytes = open(ALERT_AUDIO, "rb").read()
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio id="alarm" src="data:audio/wav;base64,{b64}"></audio>
        <script>
        function playAlarm() {{
            var a = document.getElementById('alarm');
            try {{
                a.currentTime = 0;
                a.play();
            }} catch(e) {{
                console.log("play failed", e);
            }}
        }}
        </script>
        """
        st.components.v1.html(audio_html, height=0)

# Run inference when buffer full
if len(frames_buffer) == seq_len:
    clip = torch.stack(frames_buffer).unsqueeze(0).to(device)  # [1,T,C,H,W]
    with torch.no_grad():
        out = model(clip)
        pred = torch.argmax(out, dim=1).item()
    label = "Violent" if pred == 1 else "Non-Violent"
    st.markdown(f"## Result: **{label}**")
    if pred == 1:
        st.markdown("<b style='color:red'>⚠️ Violent behavior detected!</b>", unsafe_allow_html=True)
        # attempt to play alarm via JS
        st.components.v1.html("<script>try{document.getElementById('alarm').play();}catch(e){console.log(e);}</script>", height=0)
    else:
        st.success("Normal activity - monitoring...")
else:
    st.write("Waiting for enough frames to run prediction... (التقط/حدّث الكاميرا حتى تتجمع عدد الفريمات المطلوب)")
