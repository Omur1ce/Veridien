import segmentation_models_pytorch as smp
from torchinfo import summary


def create_model(out_classes=3, in_channels=4, enc_name="resnet18", enc_weights="imagenet", batch_size=32, img_size=(224, 224)):
    model = smp.UnetPlusPlus(
        encoder_name=enc_name, 
        encoder_weights=enc_weights, 
        in_channels=1, 
        classes=3, 
    )
    summary(model, input_size=(batch_size, in_channels, *img_size), device="cpu")
    
    return model

if __name__ == "__main__":
    create_model()