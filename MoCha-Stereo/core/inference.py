"""Inference-only patches for MoCha-Stereo (any input size, no per-size compilation / tuning).

prepare_inference(model, half=True, half_acc=True, channels_last=True, kernels=True):
  - pure fp16 weights once (no autocast re-casts); BatchNorm kept as an fp32 per-channel affine applied to the conv
    output (not folded into the fp16 weights: exp 002 DESIGN_FLAWS D7)
  - layout-preserving (NHWC / NDHWC) norms: PyTorch's norm kernels return NCHW for NHWC input (exp 002 C2)
  - in-place norm / activation / residual add on fresh conv outputs (memory; same math)
  - half_acc: fp16-accumulate cuDNN convs (core/cudnn_conv.py, precompiled engines only)
  - kernels: conv epilogues (bias, BN affine, residual, activation) as one custom kernel; LSTM with one merged
    4-gate conv and fused gates; fused geometry-volume lookup (core/igev_kernels.cu, compiled once)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import submodule, extractor, update

# fused-epilogue CUTLASS convs (core/cutlass_conv.cu, port of RAFT-Stereo exp 002 v8, compiled once): GRU z|r conv +
# gates, q conv + state update, conv + bias/BN + relu for >= 128 output channels. IGEV_FUSED=0: cuDNN + separate pass.
FUSED = os.environ.get("IGEV_FUSED", "1") == "1"
_FC = [None]


def fused():
    if _FC[0] is None:
        from core.cutlass_conv import FusedConvs, EpiArgs
        _FC[0] = (FusedConvs.get(), EpiArgs)
    return _FC[0]


def fused_cfg(P):
    """Fixed size rule (results/2026-10-09_fused_probe.txt): 128x128x32/3st tile, 64x128x64/4st for small problems."""
    return 0 if P >= 7000 else 1


ACT = {"none": 0, "relu": 1, "leaky": 2, "tanh": 3, "sigmoid": 4, "relu6": 5, "silu": 6}
_K = [None]


def kern():
    if _K[0] is None:
        from core.kernels import Kernels
        _K[0] = Kernels.get()
    return _K[0]


def _mf(dim):
    return torch.channels_last if dim == 4 else torch.channels_last_3d


def _rows(x):
    """(P, C, row stride) of a dense channels-last tensor, else None."""
    if x.dim() not in (4, 5) or not x.is_contiguous(memory_format=_mf(x.dim())) or x.dtype != torch.float16:
        return None
    C = x.shape[1]
    return x.numel() // C, C, C


def epilogue(y, s=None, t=None, act=0, r=None):
    """In place: y = act(y * s + t + r), y dense channels-last fp16, s/t fp32 [C] or None, r like y or None."""
    ry = _rows(y)
    if ry is None or (r is not None and (_rows(r) is None or r.shape != y.shape)):
        return _epilogue_torch(y, s, t, act, r)
    P, C, st = ry
    k = kern()
    vec = C % 8 == 0
    name = "affine_act8" if vec else "affine_act"
    n = P * (C // 8) if vec else P * C
    k.launch(name, k.linear(n), y, st, 0 if s is None else s, 0 if t is None else t, 0 if r is None else r,
             0 if r is None else C, P, C, act)
    return y


def _epilogue_torch(y, s, t, act, r):
    shp = (1, -1) + (1,) * (y.dim() - 2)
    yf = y.float()
    if s is not None:
        yf = yf * s.view(shp)
    if t is not None:
        yf = yf + t.view(shp)
    if r is not None:
        yf = yf + r.float()
    yf = [lambda v: v, F.relu, lambda v: F.leaky_relu(v, 0.01), torch.tanh, torch.sigmoid, F.relu6][act](yf)
    return y.copy_(yf)


# ------------------------------------------------------------------------------------------------ conv + epilogue
def conv_ep(m, x, s=None, t=None, act=0, r=None, keep_pad=False):
    """conv module m (Conv/ConvTranspose 2d/3d, any groups) without bias, then one epilogue: act(conv*s + t + r).
    t defaults to the conv bias; with s (BN), pass t = BN shift + bias * s (precomputed in prepare)."""
    if t is None and s is None:
        t = m._bias32
    mf = _mf(x.dim())
    if getattr(m, "_fuse", False) and r is None and act in (0, 1) and x.dim() == 4 and x.dtype == torch.float16 \
            and x.shape[0] == 1 and x.is_contiguous(memory_format=torch.channels_last):
        y = _conv_fused(m, x, s, t, act)
        if y is not None:
            return y if y.shape[1] == m.out_channels or keep_pad else y[:, :m.out_channels]
    if getattr(m, "_hacc", False):
        from core.cudnn_conv import conv_half_acc
        y = conv_half_acc(x, m.weight, None, list(m.padding), list(m.stride))
    elif getattr(m, "_hacc_t", False):
        from core.cudnn_conv import deconv_half_acc
        y = deconv_half_acc(x, m.weight, list(m.padding), list(m.stride))
    elif isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        f = F.conv_transpose2d if x.dim() == 4 else F.conv_transpose3d
        y = f(x, m.weight, None, m.stride, m.padding, m.output_padding, m.groups, m.dilation).contiguous(memory_format=mf)
    else:
        f = F.conv2d if x.dim() == 4 else F.conv3d
        y = f(x, m.weight, None, m.stride, m.padding, m.dilation, m.groups).contiguous(memory_format=mf)
    if s is not None or t is not None or act or r is not None:
        y = epilogue(y, s, t, act, r)
    if y.shape[1] != m.out_channels and not keep_pad:  # weights zero-padded to a multiple of 8 output channels
        y = y[:, :m.out_channels]
    return y


def _conv_fused(m, x, s, t, act, out=None, out2=None):
    """conv + epilogue as one CUTLASS kernel. out / out2: (buffer, channel offset) destinations inside wider
    channels-last buffers (out2: a second copy with the same row stride)."""
    fc, E = fused()
    _, C, H, W = x.shape
    K = m.weight.shape[0]
    (R, S), (ph, pw), (st, st2) = m.kernel_size, m.padding, m.stride
    Ho, Wo = (H + 2 * ph - R) // st + 1, (W + 2 * pw - S) // st + 1
    if out is None:
        y = torch.empty((1, K, Ho, Wo), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
        yp, ys = y.data_ptr(), K
    else:
        y, c0 = out
        assert y.is_contiguous(memory_format=torch.channels_last) and tuple(y.shape[2:]) == (Ho, Wo)
        yp, ys = y.data_ptr() + 2 * c0, y.shape[1]
    y2p = None if out2 is None else out2[0].data_ptr() + 2 * out2[1]
    e = E(2, act, None, 0, yp, ys, None, ys, y2p, None, 0 if s is None else s.data_ptr(),
          0 if t is None else t.data_ptr(), None, 0, 0)
    rc = fc.run(fused_cfg(Ho * Wo), x, m.weight, H, W, C, K, e, padding=(ph, pw), stride=st)
    if rc == 1:
        m._fuse = False  # geometry not supported -> cuDNN path from now on
        return None
    if rc:
        raise RuntimeError(f"fused conv failed ({rc})")
    return y


def _conv_forward(self, x):
    return conv_ep(self, x)


def _bn_st(bn, conv):
    s = bn.weight.float() * torch.rsqrt(bn.running_var.float() + bn.eps)
    t = bn.bias.float() - bn.running_mean.float() * s
    if conv is not None and conv.bias is not None:
        t = t + conv.bias.float() * s
    return s.cuda().contiguous(), t.cuda().contiguous()


def _basic_conv_forward(self, x):  # BasicConv: conv -> BN -> LeakyReLU(0.01)
    act = ACT["leaky"] if self.relu else 0
    if self.use_bn:
        return conv_ep(self.conv, x, self._s, self._t, act)
    return conv_ep(self.conv, x, None, None, act)


def _nbasic_conv_forward(self, x):  # nets.feature.BasicConv: conv -> BN -> ReLU
    act = ACT["relu"] if self.relu else 0
    if self.use_bn:
        return conv_ep(self.conv, x, self._s, self._t, act)
    return conv_ep(self.conv, x, None, None, act)


def _hf_filter(feature):
    """Mca_Camp high-frequency filter: upstream convolves with a ones(C, C, 3, 3) / 9 kernel (zero padding), i.e. every
    output channel = 3x3 box mean (count_include_pad) of the channel SUM. Same values without the C x C x 9 conv."""
    s = feature.float().sum(1, keepdim=True)
    sm = F.avg_pool2d(s, 3, 1, 1, count_include_pad=True)
    return (feature.float() - sm).to(feature.dtype).contiguous(memory_format=torch.channels_last)


def _in_stats(x):
    var, mean = torch.var_mean(x.float(), dim=tuple(range(2, x.dim())), keepdim=True, unbiased=False)
    return mean, var


def _basic_conv_in_forward(self, x):  # BasicConv_IN: conv -> InstanceNorm -> LeakyReLU(0.01)
    act = ACT["leaky"] if self.relu else 0
    y = conv_ep(self.conv, x)
    if not self.use_in:
        return epilogue(y, None, None, act) if act else y
    mean, var = _in_stats(y)
    rstd = torch.rsqrt(var + self.IN.eps)
    return epilogue(y, rstd.view(-1).contiguous(), (-mean * rstd).view(-1).contiguous(), act)


def _in_module_forward(self, x):  # bare InstanceNorm2d(affine=False) (followed by ReLU in the Sequential)
    mean, var = _in_stats(x)
    rstd = torch.rsqrt(var + self.eps)
    return epilogue(x.contiguous(memory_format=torch.channels_last), rstd.view(-1).contiguous(),
                    (-mean * rstd).view(-1).contiguous(), 0)


def _bn_module_forward(self, x):  # standalone BN (or timm BatchNormAct2d) on an NHWC tensor
    return epilogue(x, self._s, self._t, self._act)


def _resblock_forward(self, x):  # batch-norm ResidualBlock (context encoder): relu(x + relu(bn2(conv2(relu(bn1(conv1 x))))))
    y = conv_ep(self.conv1, x, self.norm1._s, self.norm1._t, ACT["relu"])
    if self.downsample is not None:
        x = conv_ep(self.downsample[0], x, self.downsample[1]._s, self.downsample[1]._t, 0)
    o = conv_ep(self.conv2, y, self.norm2._s, self.norm2._t, ACT["relu"])
    del y
    return epilogue(o, None, None, ACT["relu"], r=x)


STRIP_PIXELS = 2 * 2**20   # context-encoder full-res stage processed in row strips of about this many pixels
STRIP_HALO = 16            # full-res rows of receptive field of conv1(7x7) + layer1 + layer2 (14), kept even


def _stem(self, x):
    x = conv_ep(self.conv1, x, self.norm1._s, self.norm1._t, ACT["relu"])
    x = self.layer1(x)
    return self.layer2(x)


def _encoder_stem_forward(self, x, dual_inp=False, num_layers=3):
    """MultiBasicEncoder with conv1 + norm1 + relu fused. The full-resolution part (conv1, layer1: 64 channels at
    input resolution, then layer2 to 1/2) runs in row strips with a halo: all ops there are local (eval BatchNorm),
    so the result is the same, but only one strip of full-res activations is alive (Midd-F: 3 x 0.78 GB -> ~0.8 GB)."""
    H, W = x.shape[-2:]
    rows = max(64, (STRIP_PIXELS // W) // 2 * 2)
    if H <= rows + 2 * STRIP_HALO or self.downsample != 2:
        x = _stem(self, x)
    else:
        out = None
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            a, b = max(0, r0 - STRIP_HALO), min(H, r1 + STRIP_HALO)
            y = _stem(self, x[:, :, a:b])
            if out is None:
                out = torch.empty((y.shape[0], y.shape[1], (H + 1) // 2, y.shape[3]), device=y.device, dtype=y.dtype,
                                  memory_format=torch.channels_last)
            t = (r0 - a) // 2
            out[:, :, r0 // 2:(r1 + 1) // 2] = y[:, :, t:t + (r1 + 1) // 2 - r0 // 2]
            del y
        x = out
    x = self.layer3(x)
    outputs04 = [f(x) for f in self.outputs04]
    if num_layers == 1:
        return (outputs04,)
    y = self.layer4(x)
    outputs08 = [f(y) for f in self.outputs08]
    if num_layers == 2:
        return (outputs04, outputs08)
    z = self.layer5(y)
    outputs16 = [f(z) for f in self.outputs16]
    return (outputs04, outputs08, outputs16)


def remp_fast(self, low_disp, left_img, right_img):
    """REMP.forward with the disparity warp in fp32 (fp16 sampling coordinates would be off by up to ~1 px at 2-3k
    width), the convolutional part in fp16 NHWC. low_disp / images fp32 at full resolution; returns fp32 (-disp)."""
    from nets.warp import disp_warp
    disp = low_disp.float()
    warped_right = disp_warp(right_img.float(), disp)[0]
    flaw = warped_right - left_img
    ref_flaw = torch.cat((flaw, left_img), dim=1).half().contiguous(memory_format=torch.channels_last)
    ref_flaw = self.conv1(ref_flaw)
    disp_fea = self.conv2(disp.half().contiguous(memory_format=torch.channels_last))
    x = torch.cat((ref_flaw, disp_fea), dim=1)
    x = self.conv_start(x)
    x = self.RefinementBlock(x)
    low = self.LFE(self.AP(x))
    motif = self.LMC(x)
    x = torch.mul((1 - motif), low) + torch.mul(motif, x)
    x = self.final_conv(x)
    return -F.relu(disp + x.float())


def prepare_context_mocha(model, inp_list):
    """inp_list[i] = relu(cnet output) -> per level [P, 4*Hd] context (bi | bf | bc | bo) + merged gate conv biases."""
    out = []
    lstms = [model.update_block.lstm04, model.update_block.lstm08, model.update_block.lstm16]
    for x, conv, g in zip(inp_list, model.context_zqr_convs, lstms):
        c = conv_ep(conv, x)
        out.append(epilogue(c.contiguous(memory_format=torch.channels_last), None, g._b4, 0))
    return out


def _hx_buf(h, parts):
    """[h | parts...] channels-last conv input (parts: tensors or ("pool2x" | "interp", src))."""
    b, Hd, hh, ww = h.shape
    k = kern()
    widths = [p.shape[1] if isinstance(p, torch.Tensor) else p[1].shape[1] for p in parts]
    ctot = Hd + sum(widths)
    hx = torch.empty((b, ctot, hh, ww), device=h.device, dtype=h.dtype, memory_format=torch.channels_last)
    hx[:, :Hd].copy_(h)
    c0 = Hd
    for p, cw in zip(parts, widths):
        if isinstance(p, torch.Tensor):
            hx[:, c0:c0 + cw].copy_(p)
        else:
            kind, src = p
            k.launch(kind + "_into", k.linear(hh * ww * cw // 8), src, cw, src.shape[2], src.shape[3],
                     hx.data_ptr() + 2 * c0, ctot, hh, ww, cw)
        c0 += cw
    return hx


def _lstm_fast(g, c, h, ctx, parts):
    """LSTM(c, h, bi, bf, bc, bo, *x): Mogrifier when x has as many channels as h (lstm16), merged 4-gate conv,
    fused gate kernel; c (fp32) and h (fp16) updated in place."""
    b, Hd, hh, ww = h.shape
    hin = h
    if len(parts) == 1 and (parts[0].shape[1] if isinstance(parts[0], torch.Tensor) else parts[0][1].shape[1]) == Hd:
        p = parts[0]
        x = p.clone() if isinstance(p, torch.Tensor) else \
            F.avg_pool2d(p[1], 3, stride=2, padding=1).contiguous(memory_format=torch.channels_last)  # == pool2x
        hin = h.clone()
        m = g.m
        for ind in range(m.iters):  # x = 2 sigmoid(Q h) x ; h = 2 sigmoid(R x) h
            if ind % 2 == 0:
                x.mul_(conv_ep(m.Q, hin, act=ACT["sigmoid"])).mul_(2.0)
            else:
                hin.mul_(conv_ep(m.R, x, act=ACT["sigmoid"])).mul_(2.0)
        parts = [x]
    hx = _hx_buf(hin, parts)
    gates = conv_ep(g.conv4, hx)
    k = kern()
    P = b * hh * ww
    k.launch("lstm_gates", k.linear(P * Hd), gates, ctx, 0 if c is None else c, h, Hd, P, Hd)
    return c, h


def update_block_mocha(self, netH, ctx, corr, disp, compute_mask):
    """LSTMMultiUpdateBlock.forward, 3 levels, inference (netH fp16 updated in place). Upstream calls it with
    netC = net_list (the same list), so `netC[i], netH[i] = lstm(...)` leaves h in the slot: the cell input of every
    step is the previous hidden state (c is never carried over). Reproduced exactly: c = None -> kernel reads h."""
    _lstm_fast(self.lstm16, None, netH[2], ctx[2], [("pool2x", netH[1])])
    _lstm_fast(self.lstm08, None, netH[1], ctx[1], [("pool2x", netH[0]), ("interp", netH[2])])
    mf = _motion_encoder_fast(self.encoder, disp, corr)
    _lstm_fast(self.lstm04, None, netH[0], ctx[0], [mf, ("interp", netH[1])])
    d = conv_ep(self.disp_head.conv1, netH[0], act=ACT["relu"])
    delta = conv_ep(self.disp_head.conv2, d)
    mask = conv_ep(self.mask_feat_4[0], netH[0], act=ACT["relu"]) if compute_mask else None
    return mask, delta


def _motion_encoder_fast(self, disp, corr):
    cor = conv_ep(self.convc1, corr, act=ACT["relu"])
    cor = conv_ep(self.convc2, cor, act=ACT["relu"])
    d = conv_ep(self.convd1, disp, act=ACT["relu"])
    d = conv_ep(self.convd2, d, act=ACT["relu"])
    out = conv_ep(self.conv, torch.cat([cor, d], 1), act=ACT["relu"], keep_pad=True)  # 127 outputs padded to 128
    if out.shape[1] == self.conv.out_channels + 1:
        out[:, -1:].copy_(disp)        # == torch.cat([out, disp], 1) without the copy
        return out
    return torch.cat([out[:, :self.conv.out_channels], disp], 1).contiguous(memory_format=torch.channels_last)


# ------------------------------------------------------------------------------------------------ prepare
def prepare_inference(model, half=True, half_acc=True, channels_last=True, kernels=True):
    model.eval()
    dt = torch.float16 if half else torch.float32
    if kernels:
        assert half and channels_last, "custom kernels are NHWC fp16"
        # MoCha LSTMs: one merged conv for the 4 gates (order i, f, c~, o; biases go into the context, prepare_context_mocha)
        for g in (model.update_block.lstm04, model.update_block.lstm08, model.update_block.lstm16):
            convs = (g.conv_it, g.conv_ft, g.conv_c_t, g.conv_ot)
            w4 = nn.Conv2d(g.conv_it.in_channels, 4 * g.conv_it.out_channels, 3, padding=1, bias=False)
            w4.weight.data = torch.cat([c.weight.data for c in convs], 0)
            g.conv4 = w4.to(g.conv_it.weight.device)
            g._b4 = torch.cat([c.bias.data.float() for c in convs]).contiguous()
        from nets.feature import BasicConv as NBasicConv
        from nets.MCCV import Mca_Camp
    for m in model.modules():
        t = type(m)
        if kernels:
            if t is submodule.BasicConv:
                if m.use_bn:
                    m._s, m._t = _bn_st(m.bn, m.conv)
                m.forward = _basic_conv_forward.__get__(m)
            elif t is NBasicConv and not isinstance(m.conv, (nn.Conv3d, nn.ConvTranspose3d)):  # conv -> BN -> ReLU
                if m.use_bn:
                    m._s, m._t = _bn_st(m.bn, m.conv)
                m.forward = _nbasic_conv_forward.__get__(m)
            elif t is Mca_Camp:
                m.high_frequency_sliding_window_filter = _hf_filter
            elif t is submodule.BasicConv_IN:
                m.forward = _basic_conv_in_forward.__get__(m)
            elif t is extractor.ResidualBlock and isinstance(m.norm1, nn.BatchNorm2d):
                m.norm1._s, m.norm1._t = _bn_st(m.norm1, m.conv1)
                m.norm2._s, m.norm2._t = _bn_st(m.norm2, m.conv2)
                if m.downsample is not None:
                    m.downsample[1]._s, m.downsample[1]._t = _bn_st(m.downsample[1], m.downsample[0])
                m.forward = _resblock_forward.__get__(m)
            elif t is extractor.MultiBasicEncoder:
                m.norm1._s, m.norm1._t = _bn_st(m.norm1, m.conv1)
                m.forward = _encoder_stem_forward.__get__(m)
            elif t.__name__ == "BatchNormAct2d" or t in (nn.BatchNorm2d, nn.BatchNorm3d):
                if not hasattr(m, "_s"):
                    m._s, m._t = _bn_st(m, None)
                an = type(getattr(m, "act", nn.Identity())).__name__
                m._act = {"Identity": 0, "ReLU": 1, "ReLU6": 5, "SiLU": 6}[an]
                m.forward = _bn_module_forward.__get__(m)
            elif t is nn.InstanceNorm2d and not m.affine and not m.track_running_stats:
                m.forward = _in_module_forward.__get__(m)
    model.to(dt)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            m._bias32 = None if m.bias is None else m.bias.float().contiguous()
        if channels_last:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)
    n = 0
    if half_acc:
        assert half
        from core.cudnn_conv import patch_convs
        n = patch_convs(model, mark_only=kernels)
    if kernels:
        for m in model.modules():
            if type(m) in (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d):
                m.forward = _conv_forward.__get__(m)
            # conv + epilogue as one CUTLASS kernel: only >= 128 output channels (exp 002 v8 rule: 64/96-channel convs
            # are as fast or faster as cuDNN + separate pass)
            m._fuse = FUSED and type(m) is nn.Conv2d and m.groups == 1 and tuple(m.dilation) == (1, 1) and \
                m.stride[0] == m.stride[1] and m.weight.shape[0] >= 128 and m.weight.shape[0] % 8 == 0 and \
                m.weight.shape[1] % 8 == 0 and m.weight.dtype == torch.float16 and \
                isinstance(m.padding, tuple) and m.padding_mode == "zeros"
    model.inf_dtype, model.inf_cl, model.inf_kernels = dt, channels_last, kernels
    return n
