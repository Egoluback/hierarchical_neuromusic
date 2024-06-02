import math

import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from miditok.midi_tokenizer import MIDITokenizer
from torch import nn
from torch.nn.modules.normalization import LayerNorm

from .music_transformer import DummyDecoder
from .music_transformer import MusicTransformer
from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderLayerRPR, TransformerEncoderRPR


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def pad_to_multiple(tensor, multiple, dim=-2, value=0):
    seq_len = tensor.shape[dim]
    m = seq_len / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def cast_tuple(val, depth=1):
    return val if isinstance(val, tuple) else ((val,) * depth)


def get_transformer_block(dim_feedforward, num_layers, d_model, p_dropout, nhead, **kwargs):
    encoder_norm = LayerNorm(d_model)
    encoder_layer = TransformerEncoderLayerRPR(d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead,
                                               p_dropout=p_dropout, **kwargs)
    encoder = TransformerEncoderRPR(encoder_layer=encoder_layer, num_layers=num_layers, norm=encoder_norm)
    return encoder


# factory

def get_hourglass_transformer(
        dim_feedforward,
        *,
        depth,
        shorten_factor,
        attn_resampling,
        updown_sample_type,
        p_dropout,
        nhead,
        **kwargs
):
    assert isinstance(depth, int) or (isinstance(depth, tuple) and len(
        depth) == 3), 'depth must be either an integer or a tuple of 3, indicating (pre_transformer_depth, <nested-hour-glass-config>, post_transformer_depth)'
    assert not (isinstance(depth,
                           int) and shorten_factor), 'there does not need to be a shortening factor when only a single transformer block is indicated (depth of one integer value)'

    if isinstance(depth, int):
        return get_transformer_block(dim_feedforward, depth, p_dropout=p_dropout, nhead=nhead, **kwargs)

    return HourglassTransformer(dim_feedforward=dim_feedforward, depth=depth, shorten_factor=shorten_factor,
                                attn_resampling=attn_resampling,
                                nhead=nhead,
                                updown_sample_type=updown_sample_type, **kwargs)


# up and down sample classes

class NaiveDownsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return reduce(x, '(n s) b d -> n b d', 'mean', s=self.shorten_factor)


class NaiveUpsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return repeat(x, 'n b d -> (n s) b d', s=self.shorten_factor)


class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, '(n s) b d -> n b (s d)', s=self.shorten_factor)
        return self.proj(x)


class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'n b (s d) -> (n s) b d', s=self.shorten_factor)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        print(x.shape)
        return self.fn(self.norm(x), **kwargs) + x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            causal=False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h, device = self.heads, x.device
        kv_input = default(context, x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device=device, dtype=torch.bool).triu_(j - i + 1)
            mask = rearrange(mask, 'i j -> () () i j')
            sim = sim.masked_fill(mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )


# transformer classes

class Transformer(nn.Module):
    def __init__(
            self,
            dim_feedforward,
            *,
            num_layers,
            causal=True,
            nhead=8,
            d_model=64,
            p_dropout=0.,
            ff_mult=4,
            ff_dropout=0.,
            norm_out=False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNormResidual(dim_feedforward,
                                Attention(dim_feedforward, heads=nhead, dim_head=d_model, dropout=p_dropout,
                                          causal=causal)),
                PreNormResidual(dim_feedforward, FeedForward(dim_feedforward, mult=ff_mult, dropout=ff_dropout))
            ]))

        self.norm = nn.LayerNorm(dim_feedforward) if norm_out else nn.Identity()

    def forward(self, x, context=None, mask=None):
        for attn, ff in self.layers:
            x = attn(x, context=context, mask=mask)
            x = ff(x)

        return self.norm(x)


# transformer class

class HourglassTransformer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward,
            depth,
            p_dropout=0.1,
            er_len=None,
            shorten_factor=2,
            attn_resampling=False,
            updown_sample_type='naive',
            causal=False
    ):
        super().__init__()
        assert len(depth) == 3, 'depth should be a tuple of length 3'
        assert updown_sample_type in {'naive',
                                      'linear'}, 'downsample / upsample type must be either naive (average pool and repeat) or linear (linear projection and reshape)'

        pre_layers_depth, valley_depth, post_layers_depth = depth

        if isinstance(shorten_factor, (tuple, list)):
            shorten_factor, *rest_shorten_factor = shorten_factor
        elif isinstance(valley_depth, int):
            shorten_factor, rest_shorten_factor = shorten_factor, None
        else:
            shorten_factor, rest_shorten_factor = shorten_factor, shorten_factor

        transformer_kwargs = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            p_dropout=p_dropout,
            er_len=er_len
        )

        self.causal = causal
        self.shorten_factor = shorten_factor

        if updown_sample_type == 'naive':
            self.downsample = NaiveDownsample(shorten_factor)
            self.upsample = NaiveUpsample(shorten_factor)
        elif updown_sample_type == 'linear':
            self.downsample = LinearDownsample(d_model, shorten_factor)
            self.upsample = LinearUpsample(d_model, shorten_factor)
        else:
            raise ValueError(f'unknown updown_sample_type keyword value - must be either naive or linear for now')

        self.valley_transformer = get_hourglass_transformer(
            shorten_factor=rest_shorten_factor,
            depth=valley_depth,
            attn_resampling=attn_resampling,
            updown_sample_type=updown_sample_type,
            **transformer_kwargs
        )

        # self.attn_resampling_pre_valley = Transformer(num_layers=1, d_model=d_model,
        #                                               nhead=nhead,
        #                                               dim_feedforward=dim_feedforward,
        #                                               p_dropout=p_dropout) if attn_resampling else None
        self.attn_resampling_pre_valley = get_transformer_block(num_layers=1,d_model=d_model,
                                                                nhead=nhead,
                                                                dim_feedforward=dim_feedforward,
                                                                p_dropout=p_dropout,
                                                                er_len=None) if attn_resampling else None
        self.attn_resampling_post_valley = get_transformer_block(num_layers=1,d_model=d_model,
                                                                 nhead=nhead,
                                                                 dim_feedforward=dim_feedforward,
                                                                 p_dropout=p_dropout,
                                                                 er_len=None) if attn_resampling else None

        self.pre_transformer = get_transformer_block(num_layers=post_layers_depth, **transformer_kwargs)
        self.post_transformer = get_transformer_block(num_layers=post_layers_depth, **transformer_kwargs)

    def forward(self, x, mask=None, src_key_padding_mask=None, is_causal=True):
        # b: batch, n: sequence length, d: feature dimension, s: shortening factor
        # print(mask.shape)

        s = self.shorten_factor
        n, b = x.shape[:2]

        # shape (n, b, d)

        # top half of hourglass, pre-transformer layers
        x = self.pre_transformer(x, mask=mask, src_key_padding_mask=src_key_padding_mask)

        # pad to multiple of shortening factor, in preparation for pooling
        x = pad_to_multiple(x, s, dim=-1)

        if exists(mask):
            padded_mask = pad_to_multiple(mask, s, dim=-2, value=False)

        # save the residual, and for "attention resampling" at downsample and upsample
        x_residual = x.clone()

        is_causal = True
        if is_causal:
            shift = s - 1
            x = F.pad(x, (0, 0, shift, -shift), value=0.)

            if exists(mask):
                padded_mask = F.pad(padded_mask, (shift, -shift), value=False)

        # downsample
        downsampled = self.downsample(x)

        if exists(mask):
            downsampled_mask = reduce(padded_mask, '(n s) b -> n b', 'sum', s=s) > 0
            downsampled_mask = reduce(downsampled_mask, 'n (s b) -> n b', 'sum', s=s) > 0
        else:
            downsampled_mask = None

        # pre-valley "attention resampling" - they have the pooled token in each bucket attend to the tokens pre-pooled
        if exists(self.attn_resampling_pre_valley):
            # TODO
            if exists(mask):
                attn_resampling_mask = downsampled_mask
            else:
                attn_resampling_mask = None

            # print(downsampled.shape, x.shape)
            # print("attn resampling")
            downsampled = downsampled + self.attn_resampling_pre_valley(
                downsampled,
                context=x,
                mask=attn_resampling_mask
            )
            # downsampled = self.attn_resampling_pre_valley(
            #     rearrange(downsampled, 'n b d -> (n b) () d'),
            #     context=rearrange(x, '(n s) b d -> (n b) s d', s = s),
            #     mask=downsampled_mask
            # )


        # the "valley" - either a regular transformer or another hourglass
        x = self.valley_transformer(downsampled, mask=downsampled_mask, src_key_padding_mask=src_key_padding_mask)

        valley_out = x.clone()

        # upsample
        x = self.upsample(x)

        # add the residual
        x = x + x_residual

        # post-valley "attention resampling"
        if exists(self.attn_resampling_post_valley):
            # print(x.shape, valley_out.shape)
            # x = self.attn_resampling_post_valley(
            #     rearrange(x, '(n s) b d -> (n b) s d', s = s),
            #     context=rearrange(valley_out, 'n b d -> (n b) () d')
            # )
            x = x + self.attn_resampling_post_valley(
                x,
                context=valley_out
            )

        # bring sequence back to original length, if it were padded for pooling
        x = x[:, :n]

        # post-valley transformers
        x = self.post_transformer(x, mask=mask, src_key_padding_mask=src_key_padding_mask)

        return x  # is already normed in RPR Transformer Encoder


class HierarchicalMusicTransformer(MusicTransformer):
    """The implementation of Music Transformer.

    Music Transformer reproduction from https://arxiv.org/abs/1809.04281.
    Arguments allow for tweaking the transformer architecture
    (https://arxiv.org/abs/1706.03762) and the rpr argument toggles
    Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class
    with DummyDecoder to make a decoder-only transformer architecture.

    For RPR support, there is modified Pytorch 1.2.0 code in model/rpr.py.
    """

    def __init__(
            self,
            tokenizer: MIDITokenizer,
            input_length: int,
            depth: tuple,
            num_heads: int = 8,
            d_model: int = 768,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
            shorten_factor=2,
            updown_sample_type: str = "naive",
            attn_resampling=False
    ) -> None:
        """Inits MusicTransformer.

        Default parameters are taken from section 4.2 of the original article:
        https://arxiv.org/abs/1809.04281

        Args:
            n_out (int): vocab size, number of probabilities to return
            n_seq (int): length of input sequence
            n_layers (int): A number of layers in the encoder.
            num_heads (int): A number of heads used in Multi-Head attention.
            d_model (int): A token embedding size.
            dim_feedforward (int): A dimension of the feedforward network model
                used in nn.Transformer.
            dropout (float): A dropout value in Positional Encoding and in
                encoder layers.
            rpr (bool): A boolean value indicating whether to use Relative
                Positional Encoding or not.
        """
        super().__init__(tokenizer, input_length)

        n_class = len(tokenizer)
        pad_id = tokenizer["PAD_None"]

        self.dummy = DummyDecoder()
        self.nhead = num_heads
        self.d_model = d_model
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.max_seq = input_length

        # Input embedding
        self.embedding = nn.Embedding(
            n_class,
            self.d_model,
            padding_idx=pad_id,
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.d_model,
            dropout=self.dropout,
            max_len=self.max_seq,
        )

        def to_tuple_rec(tup):
            res = [to_tuple_rec(i) if type(i) == list else i for i in tup]
            return tuple(res)

        depth = to_tuple_rec(depth)

        # Define encoder as None for Base Transformer
        encoder = HourglassTransformer(self.d_model, self.nhead, self.d_ff, depth, self.dropout,
                                       updown_sample_type=updown_sample_type, attn_resampling=attn_resampling,
                                       shorten_factor=shorten_factor,
                                       er_len=self.max_seq)

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=1,
            num_decoder_layers=0,
            dropout=self.dropout,
            dim_feedforward=self.d_ff,
            custom_decoder=self.dummy,
            custom_encoder=encoder
        )

        self.Wout = nn.Linear(self.d_model, n_class)
