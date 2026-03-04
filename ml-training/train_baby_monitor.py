from ultralytics import YOLO
import torch
import torch_directml

'''# Monkeypatch para corrigir "RuntimeError: Return counts not implemented for unique operator for DirectML"
_old_unique = torch.unique
def _patched_unique(input, *args, **kwargs):
    if input.device.type == 'privateuseone':
        # Move para CPU, processa e volta para o dispositivo original
        device = input.device
        res = _old_unique(input.cpu(), *args, **kwargs)
        if isinstance(res, tuple):
            return tuple(t.to(device) for t in res)
        return res.to(device)
    return _old_unique(input, *args, **kwargs)
torch.unique = _patched_unique
'''
def main():
    # Load model
    model = YOLO("yolov8n.pt")

    model.train(data="ml-training/baby_monitor.yaml", epochs=30)
    metrics = model.val()

if __name__ == "__main__":
    main()