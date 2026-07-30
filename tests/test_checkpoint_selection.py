from src.common.experiment.checkpoint import BestTracker


def test_three_validation_best_metrics_are_independent() -> None:
    tracker = BestTracker()
    assert tracker.consider(psnr=20.0, ssim=0.8, val_loss=0.2) == {
        "psnr": True,
        "ssim": True,
        "val_loss": True,
    }
    assert tracker.consider(psnr=19.0, ssim=0.9, val_loss=0.3) == {
        "psnr": False,
        "ssim": True,
        "val_loss": False,
    }
    assert tracker.consider(psnr=21.0, ssim=0.7, val_loss=0.1) == {
        "psnr": True,
        "ssim": False,
        "val_loss": True,
    }
    assert tracker.psnr == 21.0
    assert tracker.ssim == 0.9
    assert tracker.val_loss == 0.1


def test_equal_values_do_not_replace_best() -> None:
    tracker = BestTracker(psnr=20.0, ssim=0.8, val_loss=0.2)
    assert tracker.consider(psnr=20.0, ssim=0.8, val_loss=0.2) == {
        "psnr": False,
        "ssim": False,
        "val_loss": False,
    }


def test_test_metrics_never_update_validation_best() -> None:
    tracker = BestTracker(psnr=20.0, ssim=0.8, val_loss=0.2)
    assert tracker.consider(
        psnr=99.0, ssim=0.99, val_loss=0.001, source="test"
    ) == {"psnr": False, "ssim": False, "val_loss": False}
    assert (tracker.psnr, tracker.ssim, tracker.val_loss) == (20.0, 0.8, 0.2)
