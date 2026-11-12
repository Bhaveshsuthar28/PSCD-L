
import torch

from src.models.decoder import (
    ChangeDetectionDecoder,
    count_parameters,
)


def main():

    print("=" * 55)
    print("TRINETRA DECODER TEST")
    print("=" * 55)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ------------------------------------------------------
    # Create model
    # ------------------------------------------------------

    model = ChangeDetectionDecoder(
        feature_dim=768,
        feature_channels=64,
        dropout=0.05,
    ).to(device)

    print("\nMODEL PARAMETERS")

    for key, value in count_parameters(model).items():
        print(f"{key}: {value:,}")

    # ------------------------------------------------------
    # Create dummy feature tensors
    # ------------------------------------------------------

    batch_size = 2

    t0 = torch.randn(
        batch_size,
        768,
        37,
        37,
        device=device,
    )

    t1 = torch.randn(
        batch_size,
        768,
        37,
        37,
        device=device,
    )

    print("\nINPUT SHAPES")
    print("t0:", tuple(t0.shape))
    print("t1:", tuple(t1.shape))

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------

    model.train()

    logits = model(t0, t1)

    print("\nOUTPUT")
    print("Logits:", tuple(logits.shape))
    print("Logits dtype:", logits.dtype)

    assert logits.shape == (
        batch_size,
        1,
        512,
        512,
    )

    assert torch.isfinite(logits).all()

    # ------------------------------------------------------
    # Loss and backward pass
    # ------------------------------------------------------

    target = torch.randint(
        0,
        2,
        (
            batch_size,
            1,
            512,
            512,
        ),
        device=device,
    ).float()

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
    )

    loss.backward()

    print("\nGRADIENT VALIDATION")

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert gradients_exist

    print("Loss:", loss.item())
    print("Gradient flow:", gradients_exist)

    # ------------------------------------------------------
    # Parameter update test
    # ------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    optimizer.step()

    print("Optimizer step: passed")

    # ------------------------------------------------------
    # Inference test
    # ------------------------------------------------------

    model.eval()

    with torch.no_grad():
        logits = model(t0, t1)
        probabilities = torch.sigmoid(logits)

    assert probabilities.shape == (
        batch_size,
        1,
        512,
        512,
    )

    assert probabilities.min() >= 0
    assert probabilities.max() <= 1

    print("\nINFERENCE")
    print("Probability shape:", tuple(probabilities.shape))
    print("Minimum:", probabilities.min().item())
    print("Maximum:", probabilities.max().item())

    print("\n" + "=" * 55)
    print("ALL DECODER TESTS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
