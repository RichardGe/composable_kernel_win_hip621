#!/bin/sh
# call from top of CK folder
EXE=./build/bin/tile_example_layernorm2d_fwd

for pr_i in "fp16" "bf16" ; do
for fadd in "0" "1" "2"; do
$EXE -prec_i=$pr_i -fadd=$fadd -m=99  -n=13
$EXE -prec_i=$pr_i -fadd=$fadd -m=17  -n=16
$EXE -prec_i=$pr_i -fadd=$fadd -m=1   -n=100
$EXE -prec_i=$pr_i -fadd=$fadd -m=4   -n=128
$EXE -prec_i=$pr_i -fadd=$fadd -m=80  -n=127
$EXE -prec_i=$pr_i -fadd=$fadd -m=22  -n=255 -stride=256
$EXE -prec_i=$pr_i -fadd=$fadd -m=7   -n=599
$EXE -prec_i=$pr_i -fadd=$fadd -m=19  -n=512
$EXE -prec_i=$pr_i -fadd=$fadd -m=33  -n=313 -stride=1000
$EXE -prec_i=$pr_i -fadd=$fadd -m=11  -n=510
$EXE -prec_i=$pr_i -fadd=$fadd -m=171 -n=676 -stride=818
$EXE -prec_i=$pr_i -fadd=$fadd -m=91  -n=636
$EXE -prec_i=$pr_i -fadd=$fadd -m=12  -n=768 -stride=800
$EXE -prec_i=$pr_i -fadd=$fadd -m=100 -n=766 -stride=812
$EXE -prec_i=$pr_i -fadd=$fadd -m=31  -n=1024
$EXE -prec_i=$pr_i -fadd=$fadd -m=64  -n=1000 -stride=1004
$EXE -prec_i=$pr_i -fadd=$fadd -m=8   -n=1501
$EXE -prec_i=$pr_i -fadd=$fadd -m=3   -n=1826
$EXE -prec_i=$pr_i -fadd=$fadd -m=5   -n=2040
$EXE -prec_i=$pr_i -fadd=$fadd -m=7   -n=2734
$EXE -prec_i=$pr_i -fadd=$fadd -m=1   -n=3182
$EXE -prec_i=$pr_i -fadd=$fadd -m=9   -n=4096
$EXE -prec_i=$pr_i -fadd=$fadd -m=3   -n=8192
$EXE -prec_i=$pr_i -fadd=$fadd -m=1   -n=10547
$EXE -prec_i=$pr_i -fadd=$fadd -m=3   -n=17134
done
done
