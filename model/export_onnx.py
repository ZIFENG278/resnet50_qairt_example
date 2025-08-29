import torch
from torchvision.models import resnet50

input_shape = (1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
dummy_input = torch.randn(input_shape)
filename = "./resnet50.onnx"

# Load a pretrained resnet50 model in torch
pt_model = resnet50(pretrained=True)

# Export the torch model to onnx
torch.onnx.export(pt_model.eval(),
                  dummy_input,
                  filename,
                  training=torch.onnx.TrainingMode.EVAL,
                  export_params=True,
                  do_constant_folding=False,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={
                      'input' : {0 : 'batch_size'},
                      'output' : {0 : 'batch_size'},
                  }
                  )
