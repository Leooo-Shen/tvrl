#!/bin/bash
# cardiac
pretrained_weights=
python extract_embed.py pretrained_weights=$pretrained_weights data=cardiac_1c8f2s_sup use_time_embed=False


# oct 
pretrained_weights=
python extract_embed.py pretrained_weights=$pretrained_weights data=oct_1c8f_sup use_time_embed=False