# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import torch
import poptorch

from ..builder import BACKBONES
from .darknet import Darknet


class myidentity(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        add_value = x.new_tensor(1e-6)
        x = x + add_value
        x = x - add_value
        return x


@BACKBONES.register_module()
class IPUDarknet(Darknet):
    """IPUDarknet backbone.

    Args:
        serial_num (int): Input a batch of data, it will divide the data into {serial_num} 
            batches and calculate them in turn. This method is mainly used to reduce 
            the peak memory.
        kwargs (dict): See Darknet for more details.
    """

    def __init__(self,
                 serial_num,
                 **kwargs):
        self.serial_num = serial_num
        super().__init__(**kwargs)
        assert len(self.cr_blocks) == 6
        # self.identity = torch.nn.Identity()
        self.identity = myidentity()
        # poptorch.BeginBlock(self.identity, 'identity_on_bb',1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        assert batch_size % self.serial_num == 0
        local_bz = batch_size // self.serial_num

        b3_out_list = []
        b4_out_list = []
        local_input = x
        for i in range(self.serial_num):
            start_i = local_bz * i
            end_i = start_i + local_bz
            x = local_input[start_i:end_i]
            x = poptorch.block_point(x, 0, i, "Start", True)
            x = self.conv1(x)
            x = self.conv_res_block1(x)
            x = self.conv_res_block2(x)
            x = self.conv_res_block3(x)
            b3_out_list.append(x)
            x = self.conv_res_block4(x)
            x = poptorch.block_point(x, 0, i, "Stop", True)
            b4_out_list.append(x)
        # b3_out = torch.cat(b3_out_list, dim=0)

        # b4_out_list[0] = self.identity(b4_out_list[0])

        b5_out_list = []
        for i in range(self.serial_num):
            x = b4_out_list[i]
            x = poptorch.block_point(x, 1, i, "Start", True)
            x = self.conv_res_block5(x)
            x = poptorch.block_point(x, 1, i, "Stop", True)
            b5_out_list.append(x)

        b3_out = torch.cat(b3_out_list, dim=0)
        b4_out = torch.cat(b4_out_list, dim=0)
        b5_out = torch.cat(b5_out_list, dim=0)
        return b3_out, b4_out, b5_out