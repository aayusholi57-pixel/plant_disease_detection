# Model artifacts

Large model weights are intentionally excluded from Git history.

The canonical artifact is `models/plant_disease_resnet18.pth`.

The training workflow produces this file together with evaluation reports. The checkpoint contains ResNet18 weights, ordered class names, preprocessing metadata, seed, validation score and training history.

Download the GitHub Actions artifact and place the checkpoint at `models/plant_disease_resnet18.pth` for local API inference.

Do not commit large `.pth` files unless you intentionally use Git LFS or GitHub Releases.
