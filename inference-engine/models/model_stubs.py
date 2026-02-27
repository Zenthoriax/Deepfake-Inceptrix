from fallbacks.model_fallbacks.confidence_router import PredictionResult
import time

class BaseModel:
    def __init__(self, name, default_confidence=0.85, default_uncertainty=0.15):
        self.name = name
        self.default_confidence = default_confidence
        self.default_uncertainty = default_uncertainty
        
    def predict(self, input_data, timeout=2.0):
        # Simulate some processing time
        time.sleep(0.05)
        return PredictionResult(
            output=f"{self.name}_output", 
            confidence=self.default_confidence,
            uncertainty=self.default_uncertainty
        )

# 1. Core Image Deepfake Detection Models
class EfficientNetB4(BaseModel): pass
class XceptionNet(BaseModel): pass
class ConvNeXtBase(BaseModel): pass
class ResNet50(BaseModel): pass
class VisionTransformer(BaseModel): pass

# 2. Lightweight / Fast Fallback Models
class MobileNetV3Large(BaseModel): pass
class EfficientNetB0(BaseModel): pass

# 3. Temporal Deepfake Detection Models
class CNNLSTM(BaseModel): pass
class ResNet3D18(BaseModel): pass
class SlowFastNetwork(BaseModel): pass
class TimeSformer(BaseModel): pass

# 4. Frequency / Spectral Models
class ResNet18FFT(BaseModel): pass
class EfficientNetFFT(BaseModel): pass

# 5. Biological Signal Models
class DeepPhys(BaseModel): pass
class PhysNet(BaseModel): pass

# 6. Audio Deepfake Models
class Wav2Vec2_0(BaseModel): pass
class ECAPA_TDNN(BaseModel): pass
class RawNet2(BaseModel): pass

# 7. GAN Fingerprint Models
class NoiseprintCNN(BaseModel): pass
class PatchBasedCNN(BaseModel): pass

def get_all_models():
    """Returns a dictionary mapping config names to instantiated model stubs."""
    return {
        # Core
        "efficientnet_b4": EfficientNetB4("efficientnet_b4"),
        "xceptionnet": XceptionNet("xceptionnet"),
        "convnext_base": ConvNeXtBase("convnext_base"),
        "resnet50": ResNet50("resnet50"),
        "vit_b_16": VisionTransformer("vit_b_16"),
        
        # Lightweight
        "mobilenet_v3_large": MobileNetV3Large("mobilenet_v3_large", default_confidence=0.75, default_uncertainty=0.25),
        "efficientnet_b0": EfficientNetB0("efficientnet_b0", default_confidence=0.75, default_uncertainty=0.25),
        
        # Temporal
        "cnn_lstm": CNNLSTM("cnn_lstm"),
        "r3d_18": ResNet3D18("r3d_18"),
        "slowfast_network": SlowFastNetwork("slowfast_network"),
        "timesformer": TimeSformer("timesformer"),
        
        # Spectral
        "resnet18_fft": ResNet18FFT("resnet18_fft"),
        "efficientnet_fft": EfficientNetFFT("efficientnet_fft"),
        
        # Biological
        "deepphys": DeepPhys("deepphys"),
        "physnet": PhysNet("physnet"),
        
        # Audio
        "wav2vec2_0": Wav2Vec2_0("wav2vec2_0"),
        "ecapa_tdnn": ECAPA_TDNN("ecapa_tdnn"),
        "rawnet2": RawNet2("rawnet2"),
        
        # GAN Fingerprint
        "noiseprint_cnn": NoiseprintCNN("noiseprint_cnn"),
        "patch_based_cnn": PatchBasedCNN("patch_based_cnn")
    }
