"""
Train a simple CNN for MNIST digit classification and export to ONNX.
This demonstrates the training workflow described in Chapter 5.

Usage:
    python train_mnist.py

Requirements:
    pip install torch torchvision onnx
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODEL_PATH = SCRIPT_DIR / "mnist_model.pth"
WEIGHTS_PATH = SCRIPT_DIR / "mnist_weights.bin"
ONNX_PATH = SCRIPT_DIR / "mnist_model.onnx"


class MNISTNet(nn.Module):
    """Simplest possible network for MNIST - single hidden layer.

    This is a "hello world" intro to ML inference with Vulkan.
    Architecture: 784 -> 128 -> 10 (just input->hidden->output)
    """

    def __init__(self):
        super(MNISTNet, self).__init__()

        # Single hidden layer - as simple as it gets!
        self.fc1 = nn.Linear(28 * 28, 128)  # 784 -> 128
        self.fc2 = nn.Linear(128, 10)       # 128 -> 10
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten input
        x = x.view(-1, 28 * 28)

        # Single hidden layer with ReLU
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def train_mnist(
    num_epochs=5, batch_size=64, learning_rate=0.001, data_dir=DATA_DIR
):
    """Train the MNIST model."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Create model
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, '
                      f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        train_accuracy = 100. * correct / total
        avg_loss = running_loss / len(train_loader)

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += (pred == target).sum().item()
                test_total += target.size(0)

        test_accuracy = 100. * test_correct / test_total

        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'  Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'  Test Accuracy: {test_accuracy:.2f}%\n')

    return model


def export_weights_binary(model, filename=WEIGHTS_PATH):
    """Export model weights to a simple binary format for C++ loading."""
    import struct

    model.eval()

    filename = Path(filename).expanduser().resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open('wb') as f:
        # Write magic number and version
        f.write(struct.pack('I', 0x4D4E5354))  # 'MNST'
        f.write(struct.pack('I', 1))  # Version 1

        # Helper to write tensor
        def write_tensor(tensor, name):
            data = tensor.detach().cpu().numpy().flatten().astype('float32')
            f.write(struct.pack('I', len(data)))  # Number of elements
            f.write(data.tobytes())
            print(f"  {name}: {len(data)} floats")

        # Write weights in order expected by C++ code
        # For FC-only model, we write dummy conv layers for backward compatibility
        import torch
        dummy = torch.zeros(1)
        write_tensor(dummy, "conv1.weight (dummy)")
        write_tensor(dummy, "conv1.bias (dummy)")
        write_tensor(dummy, "conv2.weight (dummy)")
        write_tensor(dummy, "conv2.bias (dummy)")

        write_tensor(model.fc1.weight, "fc1.weight")
        write_tensor(model.fc1.bias, "fc1.bias")
        write_tensor(model.fc2.weight, "fc2.weight")
        write_tensor(model.fc2.bias, "fc2.bias")
        # Dummy fc3 for compatibility
        write_tensor(dummy, "fc3.weight (dummy)")
        write_tensor(dummy, "fc3.bias (dummy)")

    print(f"\nWeights exported to {filename}")


def export_to_onnx(model, filename=ONNX_PATH):
    """Export the trained model to ONNX format."""

    import onnx

    filename = Path(filename).expanduser().resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 1, 28, 28)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=9,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported to {filename}")

    # Verify the export
    onnx_model = onnx.load(filename)
    
    # Manually downgrade IR version if it's too high for some runtimes
    if onnx_model.ir_version > 9:
        print(f"  Downgrading IR version from {onnx_model.ir_version} to 9 for compatibility")
        onnx_model.ir_version = 9
        onnx.save(onnx_model, filename)

    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")

    # Print model info
    print(f"\nONNX Model Info:")
    print(f"  Opset version: {onnx_model.opset_import[0].version}")
    print(f"  Producer: {onnx_model.producer_name}")


def test_inference(model, data_dir=DATA_DIR):
    """Test a single inference to verify the model works."""

    model.eval()

    # Get a single test image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        root=str(data_dir), train=False, download=False, transform=transform
    )
    test_image, test_label = test_dataset[0]

    # Run inference
    with torch.no_grad():
        output = model(test_image.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()

    print(f"\nTest Inference:")
    print(f"  True label: {test_label}")
    print(f"  Predicted: {predicted_class}")
    print(f"  Confidence: {probabilities[0][predicted_class].item() * 100:.2f}%")
    print(f"\nAll class probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"  {i}: {prob.item() * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MNIST and export artifacts for the C++ inference tests."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--onnx", type=Path, default=ONNX_PATH)
    parser.add_argument(
        "--skip-onnx", action="store_true", help="Do not create the optional ONNX file."
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main():
    args = parse_args()
    print("=" * 60)
    print("MNIST Training and Export")
    print("=" * 60)

    # Train model
    model = train_mnist(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
    )

    # Save PyTorch model
    model_path = args.model.expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nPyTorch model saved to {model_path}")

    # Test inference
    test_inference(model, args.data_dir)

    # Export weights for C++ inference engine
    print("\nExporting weights for C++ inference engine:")
    export_weights_binary(model, args.weights)

    # Export to ONNX (optional - requires onnxscript)
    if not args.skip_onnx:
        try:
            export_to_onnx(model, args.onnx)
        except ImportError as e:
            print(f"\nNote: ONNX export skipped (missing dependency: {e})")
            print("This is optional - the C++ weights file was exported successfully.")
        except Exception as e:
            print(f"\nWarning: ONNX export failed: {e}")
            print("This is optional - the C++ weights file was exported successfully.")

    print("\n" + "=" * 60)
    print("Training and export complete!")
    print("=" * 60)
    print("\nYou can now use:")
    print(f"  - {args.weights.expanduser().resolve()} for the C++ unit tests")
    if not args.skip_onnx:
        print(f"  - {args.onnx.expanduser().resolve()} for visualization")


if __name__ == "__main__":
    main()
