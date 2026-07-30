"""The verified project-specific NAFNet-small research configuration.

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


def _validate_fixed_integer(name: str, value: int, expected: int) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(
            f"NAFNet-small has a fixed {name}={expected}; received {value!r}."
        )
    return value


def _validate_fixed_blocks(
    name: str, value: Iterable[int], expected: Tuple[int, ...]
) -> Tuple[int, ...]:
    try:
        blocks = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"NAFNet-small has fixed {name}={list(expected)}; received "
            f"non-iterable {value!r}."
        ) from exc
    if any(type(block) is not int for block in blocks) or blocks != expected:
        raise ValueError(
            f"NAFNet-small has fixed {name}={list(expected)}; received {value!r}."
        )
    return blocks


class NAFNetSmall(NAFNet):
    """Ordinary vendored NAFNet fixed to the UIE3 three-scale configuration."""

    def __init__(
        self,
        img_channel: int = NAFNET_SMALL_IMG_CHANNEL,
        width: int = NAFNET_SMALL_WIDTH,
        enc_blk_nums: Iterable[int] = NAFNET_SMALL_ENC_BLK_NUMS,
        middle_blk_num: int = NAFNET_SMALL_MIDDLE_BLK_NUM,
        dec_blk_nums: Iterable[int] = NAFNET_SMALL_DEC_BLK_NUMS,
    ) -> None:
        img_channel = _validate_fixed_integer(
            "img_channel", img_channel, NAFNET_SMALL_IMG_CHANNEL
        )
        width = _validate_fixed_integer("width", width, NAFNET_SMALL_WIDTH)
        enc_blocks = _validate_fixed_blocks(
            "enc_blk_nums", enc_blk_nums, NAFNET_SMALL_ENC_BLK_NUMS
        )
        middle_blk_num = _validate_fixed_integer(
            "middle_blk_num", middle_blk_num, NAFNET_SMALL_MIDDLE_BLK_NUM
        )
        dec_blocks = _validate_fixed_blocks(
            "dec_blk_nums", dec_blk_nums, NAFNET_SMALL_DEC_BLK_NUMS
        )
        super().__init__(
            img_channel=img_channel,
            width=width,
            enc_blk_nums=list(enc_blocks),
            middle_blk_num=middle_blk_num,
            dec_blk_nums=list(dec_blocks),
        )
        if self.padder_size != NAFNET_SMALL_PADDER_SIZE:
            raise RuntimeError(
                "NAFNet-small must use three encoder scales and pad to a "
                f"multiple of {NAFNET_SMALL_PADDER_SIZE}; got {self.padder_size}."
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
    """Build the fixed v1 NAFNet-small without loading any weights."""

    if type != "nafnet_small":
        raise ValueError(f"v1 model type must be 'nafnet_small', got {type!r}.")
    return NAFNetSmall(
        img_channel=img_channel,
        width=width,
        enc_blk_nums=enc_blk_nums,
        middle_blk_num=middle_blk_num,
        dec_blk_nums=dec_blk_nums,
    )
