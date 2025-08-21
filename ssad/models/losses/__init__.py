#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 21:01
# @Author : WeiHua

from .rotated_sstg_dt_loss import RotatedSingleStageDTLoss  # 无标签损失
from .rotated_dense_teacher_loss import RotatedDTLoss  # 这个损失函数主要包括分类损失（loss_cls）、边界框损失（loss_bbox）、中心度损失（loss_centerness）和额外的质量-焦点损失（QFLv2）。
