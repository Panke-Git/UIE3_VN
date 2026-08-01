"""The project-specific configurable NAFNet-small implementation.

NAFNet-small is not an official upstream model name and this module does not
load pretrained weights. The inherited vendored ``NAFNet.forward`` is used
directly, including its single global residual, right/bottom zero padding, and
final crop. This wrapper adds no residual, sigmoid, or clamp.
"""

from typing import Iterable, Tuple

from third_party.nafnet.nafnet_arch import NAFNet


NAFNET_SMALL_IMG_CHANNEL = 3
NAFNET_SMALL_WIDTH = 32
NAFNET_SMALL_ENC_BLK_NUMS = (2, 2, 2)
NAFNET_SMALL_MIDDLE_BLK_NUM = 4
NAFNET_SMALL_DEC_BLK_NUMS = (2, 2, 2)
NAFNET_SMALL_PADDER_SIZE = 8


def _validate_integer(name: str, value: int, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return value


def _validate_blocks(name: str, value: Iterable[int]) -> Tuple[int, ...]:
    try:
        blocks = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty integer sequence.") from exc
    if not blocks or any(type(block) is not int or block < 0 for block in blocks):
        raise ValueError(f"{name} values must be integers >= 0, got {value!r}.")
    return blocks


class NAFNetSmall(NAFNet):
    """Ordinary vendored NAFNet with the verified small setup as defaults."""

    def __init__(
        self,
        img_channel: int = NAFNET_SMALL_IMG_CHANNEL,
        width: int = NAFNET_SMALL_WIDTH,
        enc_blk_nums: Iterable[int] = NAFNET_SMALL_ENC_BLK_NUMS,
        middle_blk_num: int = NAFNET_SMALL_MIDDLE_BLK_NUM,
        dec_blk_nums: Iterable[int] = NAFNET_SMALL_DEC_BLK_NUMS,
    ) -> None:
        img_channel = _validate_integer("img_channel", img_channel, minimum=1)
        if img_channel != 3:
            raise ValueError("NAFNet-small requires img_channel=3 for RGB data.")
        width = _validate_integer("width", width, minimum=1)
        enc_blocks = _validate_blocks("enc_blk_nums", enc_blk_nums)
        middle_blk_num = _validate_integer(
            "middle_blk_num", middle_blk_num, minimum=0
        )
        dec_blocks = _validate_blocks("dec_blk_nums", dec_blk_nums)
        if len(enc_blocks) != len(dec_blocks):
            raise ValueError(
                "enc_blk_nums and dec_blk_nums must have equal lengths."
            )
        super().__init__(
            img_channel=img_channel,
            width=width,
            enc_blk_nums=list(enc_blocks),
            middle_blk_num=middle_blk_num,
            dec_blk_nums=list(dec_blocks),
        )
        expected_padder_size = 2 ** len(enc_blocks)
        if self.padder_size != expected_padder_size:
            raise RuntimeError(
                "NAFNet-small padder size does not match its encoder depth: "
                f"expected {expected_padder_size}, got {self.padder_size}."
            )


def build_nafnet_small(
    *,
    type: str = "nafnet_small",
    img_channel: int = NAFNET_SMALL_IMG_CHANNEL,
    width: int = NAFNET_SMALL_WIDTH,
    enc_blk_nums: Iterable[int] = NAFNET_SMALL_ENC_BLK_NUMS,
    middle_blk_num: int = NAFNET_SMALL_MIDDLE_BLK_NUM,
    dec_blk_nums: Iterable[int] = NAFNET_SMALL_DEC_BLK_NUMS,
) -> NAFNetSmall:
    """Build a v1 NAFNet-small from YAML-controlled architecture values."""

    if type != "nafnet_small":
        raise ValueError(f"v1 model type must be 'nafnet_small', got {type!r}.")
    return NAFNetSmall(
        img_channel=img_channel,
        width=width,
        enc_blk_nums=enc_blk_nums,
        middle_blk_num=middle_blk_num,
        dec_blk_nums=dec_blk_nums,
    )
