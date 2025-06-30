import torch
import torchaudio

class SpeechAnalysisModel(torch.nn.Module):
    def __init__(self):
        super(SpeechAnalysisModel, self).__init__()
        # Define the same structure as when you trained it
        self.layer = torch.nn.Linear(100, 4)  # Example

    def forward(self, x):
        return self.layer(x)

def load_model(path='trained_speech_analysis_model.pth'):
    model = SpeechAnalysisModel()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    # Apply your preprocessing (e.g., MFCC, resample, normalize, etc.)
    # This should match your training pipeline
    return waveform.mean(dim=0).unsqueeze(0)  # Dummy preprocessing

def analyze_audio(model, audio_tensor):
    with torch.no_grad():
        output = model(audio_tensor)
        return {
            "confidence": round(output[0][0].item() * 100, 2),
            "clarity": round(output[0][1].item() * 100, 2),
            "emotion": "neutral" if output[0][2] < 0.5 else "excited",
            "filler": "low" if output[0][3] < 0.5 else "high"
        }
